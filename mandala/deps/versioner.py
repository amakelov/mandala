import typing
from collections import OrderedDict
import textwrap

from ..common_imports import *
from ..utils import is_subdict
from ..config import Config
from .utils import DepKey, hash_dict
from .model import (
    Node,
    CallableNode,
    GlobalVarNode,
    TerminalNode,
    DependencyGraph,
)
from .crawler import crawl_static
from .tracers import TracerABC
from ..viz import _get_colorized_diff

if Config.has_rich:
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.console import Group

from .shallow_versions import DAG
from .deep_versions import Version


class CodeState:
    def __init__(self, nodes: Dict[DepKey, Node]):
        self.nodes = nodes

    def __repr__(self) -> str:
        lines = []
        for dep_key, node in self.nodes.items():
            lines.append(f"{dep_key}:")
            lines.append(f"{node.content()}")
        return "\n".join(lines)

    def get_content_version(self, support: Iterable[DepKey]) -> str:
        return hash_dict({k: self.nodes[k].content_hash for k in support})

    def add_globals_from(self, graph: DependencyGraph):
        for node in graph.nodes.values():
            if isinstance(node, GlobalVarNode) and node.key not in self.nodes:
                self.nodes[node.key] = node


class Versioner:
    def __init__(
        self,
        paths: List[Path],
        TracerCls: type,
        strict: bool,
        track_globals: bool,
        skip_unhashable_globals: bool,
        skip_globals_silently: bool,
        skip_missing_deps: bool,
        skip_missing_silently: bool,
        track_methods: bool,
        package_name: Optional[str] = None,
    ):
        assert len(paths) in [0, 1]
        self.paths = paths
        self.TracerCls = TracerCls
        self.strict = strict

        self.skip_missing_deps = skip_missing_deps
        self.skip_missing_silently = skip_missing_silently

        self.track_globals = track_globals
        self.skip_unhashable_globals = skip_unhashable_globals
        self.skip_globals_silently = skip_globals_silently

        self.allow_methods = track_methods
        self.package_name = package_name
        self.global_topology: DependencyGraph = DependencyGraph()
        self.nodes: Dict[DepKey, Node] = {}
        self.component_dags: Dict[DepKey, DAG] = {}
        # all versions here must be synced with the DAGs already
        self.versions: Dict[DepKey, Dict[str, Version]] = {}
        columns = [
            "pre_call_uid",
            "semantic_version",
            "content_version",
            "outputs",
        ]
        self.df = pd.DataFrame(columns=columns)
    
    def drop_semantic_version(self, semantic_version: str):
        # drop from df
        self.df = self.df[self.df["semantic_version"] != semantic_version]
        # drop from .versions
        for dep_key, versions in self.versions.items():
            keys_to_drop = []
            for k, version in versions.items():
                if version.semantic_version == semantic_version:
                    keys_to_drop.append(k)
            for k in keys_to_drop:
                del versions[k]
        # TODO: we should clean up more thoroughly, but this is a start
        # TODO: remove from component_dags if no longer used
        # TODO: remove from nodes if no longer used

    def get_version_ids(
        self,
        pre_call_uid: str,
        tracer_option: Optional[TracerABC],
        is_recompute: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the content and semantic IDs for the version corresponding to the
        given pre-call uid.

        Inputs:
            - `is_recompute`: this should be true only if this is a call with
            transient outputs that we already computed once.
        """
        assert tracer_option is not None
        version = self.process_trace(
            graph=tracer_option.graph,
            pre_call_uid=pre_call_uid,
            outputs=None,
            is_recompute=is_recompute,
        )
        content_version = version.content_version
        semantic_version = version.semantic_version
        return content_version, semantic_version

    def update_global_topology(self, graph: DependencyGraph):
        for node in graph.nodes.values():
            if isinstance(node, (CallableNode, GlobalVarNode)):
                self.global_topology.add_node(node)
        for edge in graph.edges:
            if (
                edge[0] in self.global_topology.nodes.keys()
                and edge[1] in self.global_topology.nodes.keys()
            ):
                self.global_topology.edges.add(edge)

    def make_tracer(self) -> TracerABC:
        return self.TracerCls(
            paths=[Config.mandala_path] + self.paths,
            strict=self.strict,
            allow_methods=self.allow_methods,
            track_globals=self.track_globals,
            skip_unhashable_globals=self.skip_unhashable_globals,
            skip_globals_silently=self.skip_globals_silently,
        )

    def guess_code_state(self) -> CodeState:
        result_graph = DependencyGraph()
        fallback_result = {}
        for dep_key in self.global_topology.nodes.keys():
            node = self.global_topology.nodes[dep_key]
            if (
                isinstance(node, (GlobalVarNode, CallableNode))
                and dep_key not in result_graph.nodes.keys()
            ):
                obj = node.load_obj(skip_missing=self.skip_missing_deps, skip_silently=self.skip_missing_silently)
                fallback_result[dep_key] = node.from_obj(obj=obj, dep_key=dep_key)
        nodes = {**result_graph.nodes, **fallback_result}
        # fill in the gaps with a static crawl
        static_result, objs = crawl_static(
            root=None if len(self.paths) == 0 else self.paths[0],
            strict=self.strict,
            package_name=self.package_name,
            include_methods=self.allow_methods,
        )
        for dep_key, node in static_result.items():
            if dep_key not in nodes.keys():
                nodes[dep_key] = node
        result = CodeState(nodes=nodes)
        return result

    def get_codestate_semantic_hashes(
        self,
        code_state: CodeState,
    ) -> Optional[Dict[DepKey, str]]:
        """
        Given a code state, return the semantic hashes of the components found
        in this code state that *also* appear in the global component topology,
        or None if the code state is not fully compatible with the commits we
        have in the DAGs.
        """
        result = {}
        if not self.global_topology.nodes.keys() <= code_state.nodes.keys():
            extra_keys = self.global_topology.nodes.keys() - code_state.nodes.keys()
            raise ValueError(
                f"Found extra keys in global topology not in code state: {extra_keys}."
            )
        for component in code_state.nodes.keys():
            if component in self.global_topology.nodes.keys():
                component_content_hash = code_state.nodes[component].content_hash
                dag = self.component_dags[component]
                if component_content_hash not in dag.commits:
                    print(
                        f"Could not find commit for {component} with content hash {component_content_hash}"
                    )
                    return None
                result[component] = dag.commits[component_content_hash].semantic_hash
        return result

    def apply_state_hypothesis(
        self, hypothesis: CodeState, trace_result: Dict[DepKey, Node]
    ):
        keys_to_remove = []
        for trace_dep_key, trace_node in trace_result.items():
            if isinstance(trace_node, TerminalNode):
                continue
            if trace_dep_key not in hypothesis.nodes.keys() and isinstance(
                trace_node, GlobalVarNode
            ):
                continue
            if trace_dep_key in hypothesis.nodes.keys():
                if isinstance(trace_node, GlobalVarNode):
                    if (
                        not hypothesis.nodes[trace_dep_key].content()
                        == trace_node.content()
                    ):
                        print(f"Content mismatch for {trace_dep_key}")
                        print(f"Expected: {hypothesis.nodes[trace_dep_key].content()}")
                        print(f"Actual: {trace_node.content()}")
                        print(
                            f"Diff:\n{_get_colorized_diff(hypothesis.nodes[trace_dep_key].content(), trace_node.content())}"
                        )
                        raise ValueError(f"Content mismatch for {trace_dep_key}")
                elif isinstance(trace_node, CallableNode):
                    runtime_description_hypothesis = hypothesis.nodes[
                        trace_dep_key
                    ].runtime_description
                    if runtime_description_hypothesis != trace_node.runtime_description:
                        if self.strict:
                            raise ValueError(
                                f"Bytecode mismatch for {trace_dep_key} (you probably need to re-import the module)"
                            )
                    trace_node.representation = hypothesis.nodes[
                        trace_dep_key
                    ].representation
                else:
                    continue
            else:
                if self.strict:
                    raise ValueError(
                        f"Unexpected dependency {trace_dep_key}; expected {textwrap.shorten(str(hypothesis.nodes.keys()), width=80)}"
                    )
                else:
                    keys_to_remove.append(trace_dep_key)
        for key in keys_to_remove:
            del trace_result[key]

    def get_semantic_version(
        self, semantic_hashes: Dict[DepKey, str], support: Iterable[DepKey]
    ) -> str:
        return hash_dict({k: semantic_hashes[k] for k in support})

    def init_component(self, component: DepKey, node: Node, initial_content: str):
        """
        Initialize a new component with an initial state.
        """
        if isinstance(node, CallableNode):
            content_type = "code"
        elif isinstance(node, GlobalVarNode):
            content_type = "global_variable"
        else:
            raise ValueError(f"Unexpected node type {type(node)}")
        dag = DAG(content_type=content_type)
        dag.init(initial_content=initial_content)
        self.nodes[component] = node
        self.component_dags[component] = dag
        self.versions[component] = {}

    def sync_codebase(self, code_state: CodeState):
        """
        Sync all the known components from the current state of the codebase.
        """
        dags = copy.deepcopy(self.component_dags)
        for component, dag in dags.items():
            content = code_state.nodes[component].content()
            content_hash = code_state.nodes[component].content_hash
            if content_hash not in dag.commits.keys() and dag.head is not None:
                dependent_versions = self.get_dependent_versions(
                    dep_key=component, commit=dag.head
                )
                dependent_versions_presentation = textwrap.indent(
                    text="\n".join([v.presentation for v in dependent_versions]),
                    prefix="  ",
                )
                print(f"CHANGE DETECTED in {component[1]} from module {component[0]}")
                print(f"Dependent components:\n{dependent_versions_presentation}")
                # print(f"===DIFF===:")
            dag.sync(content=content)
        # update the DAGs if all commits succeeded
        self.component_dags = dags

    def sync_component(
        self,
        component: DepKey,
        is_semantic_change: Optional[bool],
        code_state: CodeState,
    ) -> str:
        """
        Sync a single component from the current state of the codebase. Useful
        as a low-level API for testing.
        """
        commit = self.component_dags[component].sync(
            content=code_state.nodes[component].content(),
            is_semantic_change=is_semantic_change,
        )
        return commit

    def get_current_versions(
        self, component: DepKey, code_state: CodeState
    ) -> List[Version]:
        code_semantic_hashes = self.get_codestate_semantic_hashes(code_state=code_state)
        result = []
        if code_semantic_hashes is None:
            return result
        for _, version in self.versions[component].items():
            if is_subdict(version.semantic_expansion, code_semantic_hashes):
                result.append(version)
        return result

    def get_semantically_compatible_versions(
        self, component: DepKey, code_state: CodeState
    ) -> List[Version]:
        code_semantic_hashes = self.get_codestate_semantic_hashes(code_state=code_state)
        if code_semantic_hashes is None:
            return []
        result = []
        for version in self.versions[component].values():
            if all(
                [
                    version.semantic_expansion[dep_key] == code_semantic_hashes[dep_key]
                    for dep_key in version.semantic_expansion.keys()
                ]
            ):
                result.append(version)
        return result

    ############################################################################
    ### processing traces
    ############################################################################
    def create_new_components_from_nodes(self, nodes: Dict[DepKey, Node]):
        """
        Given the result of a trace, create any components necessary.
        """
        ### new components must be found among the nodes in the trace result
        for dep_key, node in nodes.items():
            if dep_key not in self.nodes and not isinstance(node, TerminalNode):
                content = node.content()
                self.init_component(
                    component=dep_key, node=node, initial_content=content
                )

    def sync_version(self, version: Version, require_exists: bool = False) -> Version:
        # TODO - this is impure
        version.sync(component_dags=self.component_dags, all_versions=self.versions)
        if version.content_version not in self.versions[version.component]:
            if require_exists:
                raise ValueError(f"Version {version} does not exist in VersioningState")
            # logging.info(f'Adding new version for {version.component}')
            self.versions[version.component][version.content_version] = version
        return version

    def lookup_call(
        self, component: DepKey, pre_call_uid: str, code_state: CodeState
    ) -> Optional[Tuple[str, str]]:
        """
        Return a tuple of (content_version, semantic_version), or None if the
        call is not found.

        Inputs:
            - `pre_call_uid`: this is a hash of the content IDs of the inputs,
            together with the function's name.

        This works as follows:
        - we figure out the semantic hashes (i.e. shallow semantic versions) of
        the elements of the code state present in the global topology we have on
        record
        - we restrict to the records that match the given `pre_call_uid`
        - we search among these 
        """
        codebase_semantic_hashes = self.get_codestate_semantic_hashes(
            code_state=code_state
        )
        if codebase_semantic_hashes is None:
            return None
        candidates = self.df[self.df["pre_call_uid"] == pre_call_uid]
        if len(candidates) == 0:
            return None
        else:
            content_versions = candidates["content_version"].values.tolist()
            semantic_versions = candidates["semantic_version"].values.tolist()
            for content_version, semantic_version in zip(
                content_versions, semantic_versions
            ):
                version = self.versions[component][content_version]
                codebase_semantic = self.get_semantic_version(
                    semantic_hashes=codebase_semantic_hashes, support=version.support
                )
                if codebase_semantic == semantic_version:
                    return content_version, semantic_version
            return None

    def process_trace(
        self,
        graph: DependencyGraph,
        pre_call_uid: str,
        outputs: Any,
        is_recompute: bool,
    ) -> Version:
        component, nodes = graph.get_trace_state()
        self.create_new_components_from_nodes(nodes=nodes)
        version = Version.from_trace(
            component=component, nodes=nodes, strict=self.strict
        )
        version = self.sync_version(version=version)
        row = {
            "pre_call_uid": pre_call_uid,
            "semantic_version": version.semantic_version,
            "content_version": version.content_version,
            "outputs": outputs,
        }
        if not is_recompute:
            self._check_semantic_distinguishability(
                component=component, pre_call_uid=pre_call_uid, call_version=version
            )
            # logging.info(f"Adding new call for {pre_call_uid} for {component}")
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        return version

    def _check_semantic_distinguishability(
        self, component: DepKey, pre_call_uid: str, call_version: Version
    ):
        ### check semantic distinguishability between calls
        # TODO: make this more efficient
        candidates = self.df[self.df["pre_call_uid"] == pre_call_uid]
        existing_semantic_expansions = {}
        for content_version, semantic_version in zip(
            candidates["content_version"].values.tolist(),
            candidates["semantic_version"].values.tolist(),
        ):
            if semantic_version not in existing_semantic_expansions.keys():
                existing_call_version = self.versions[component][content_version]
                existing_semantic_expansions[
                    semantic_version
                ] = existing_call_version.semantic_expansion
        new_semantic_dep_hashes = call_version.semantic_expansion
        for semantic_version, semantic_hashes in existing_semantic_expansions.items():
            overlap = set(semantic_hashes.keys()).intersection(
                set(new_semantic_dep_hashes.keys())
            )
            if all([semantic_hashes[k] == new_semantic_dep_hashes[k] for k in overlap]):
                raise ValueError(
                    f"Call to {component} with pre_call_uid={pre_call_uid} is not semantically distinguishable from call for semantic version {semantic_version}"
                )

    ############################################################################
    ### inspecting the state
    ############################################################################
    def get_flat_versions(self) -> Dict[str, Version]:
        return {
            k: v
            for component, versions in self.versions.items()
            for k, v in versions.items()
        }

    def get_dependent_versions(self, dep_key: DepKey, commit: str) -> List[Version]:
        """
        Get a list of versions of components dependent on a given commit to a
        given component
        """
        dep_semantic = self.component_dags[dep_key].commits[commit].semantic_hash
        return [
            version
            for version in self.get_flat_versions().values()
            if version.semantic_expansion.get(dep_key) == dep_semantic
        ]

    def present_dependencies(
        self,
        commits: Dict[DepKey, str],
        include_metadata: bool = True,
        header: Optional[str] = None,
    ) -> str:
        """
        Get a code snippet for a given state of some dependencies
        """
        result_lines = []
        if header is not None:
            result_lines.extend(header.splitlines())
        module_groups = self.get_canonical_groups(components=commits.keys())
        for module_name, components_in_module in module_groups.items():
            result_lines.append(80 * "#")
            result_lines.append(f'### IN MODULE "{module_name}"')
            result_lines.append(80 * "#")
            commits_in_module = {k: commits[k] for k in components_in_module}
            nodes = {k: self.nodes[k] for k in commits_in_module.keys()}
            semantic_hashes = {
                k: self.component_dags[k].commits[v].semantic_hash
                for k, v in commits_in_module.items()
            }
            global_keys = {k for k, v in nodes.items() if isinstance(v, GlobalVarNode)}
            callable_keys = {k for k, v in nodes.items() if isinstance(v, CallableNode)}
            metadatas = {
                k: f"### {nodes[k].present_key()}\n### content_commit={commits_in_module[k]}\n### semantic_commit={semantic_hashes[k]}"
                for k in commits_in_module.keys()
            }
            for global_key in sorted(global_keys):
                if include_metadata:
                    result_lines.append(metadatas[global_key])
                global_name = global_key[1]
                result_lines.append(
                    f"{global_name} = {self.component_dags[global_key].get_presentable_content(commits_in_module[global_key])}"
                )
            result_lines.append("")
            callable_keys = list(sorted(callable_keys))
            is_class_start = []
            for i, callable_key in enumerate(callable_keys):
                this_is_class = "." in callable_key[1]
                if i == 0 and this_is_class:
                    is_class_start.append(True)
                    continue
                prev_is_class = "." in callable_keys[i - 1][1]
                if this_is_class and not prev_is_class:
                    is_class_start.append(True)
                    continue
                if (
                    this_is_class
                    and prev_is_class
                    and callable_keys[i - 1][1].rsplit(".", maxsplit=1)[0]
                    != callable_key[1].rsplit(".", maxsplit=1)[0]
                ):
                    is_class_start.append(True)
                    continue
                is_class_start.append(False)
            for i, callable_key in enumerate(callable_keys):
                if is_class_start[i]:
                    result_lines.append(
                        f"### in class {callable_key[1].rsplit('.', 1)[0]}:"
                    )
                if include_metadata:
                    result_lines.append(metadatas[callable_key])
                result_lines.append(
                    self.component_dags[callable_key].get_presentable_content(
                        commits_in_module[callable_key]
                    )
                )
                result_lines.append("")
        return "\n".join(result_lines)

    def show_versions(
        self,
        component: DepKey,
        only_semantic: bool = False,
        include_metadata: bool = True,
        plain: bool = False,
    ):
        versions_dict = self.versions[component]
        if only_semantic:
            # use just 1 semantic representative per content version
            versions = list(
                {v.semantic_version: v for v in versions_dict.values()}.values()
            )
        else:
            versions = list(versions_dict.values())
        if Config.has_rich and not plain:
            version_panels: List[Panel] = []
            for version in versions:
                header_lines = [
                    f"### Dependencies for version of {self.nodes[component].present_key()}"
                ]
                header_lines.append(f"### content_version_id={version.content_version}")
                header_lines.append(
                    f"### semantic_version_id={version.semantic_version}\n\n"
                )
                version_panels.append(
                    Panel(
                        Syntax(
                            self.present_dependencies(
                                header="\n".join(header_lines),
                                commits=version.semantic_expansion,
                                include_metadata=include_metadata,
                            ),
                            lexer="python",
                            theme="solarized-light",
                        ),
                        title=None,
                        expand=True,
                    )
                )
            rich.print(Group(*version_panels))
        else:
            for version in versions:
                print(version.presentation)
                print(
                    textwrap.indent(
                        self.present_dependencies(
                            commits=version.semantic_expansion,
                            include_metadata=include_metadata,
                        ),
                        prefix="    ",
                    )
                )

    def get_canonical_groups(
        self, components: Iterable[DepKey]
    ) -> typing.OrderedDict[str, List[DepKey]]:
        """
        Order components by module name alphabetically, and within each module,
        put the global variables first, then the callables.
        """
        result = OrderedDict()
        for component in components:
            module_name = component[0]
            if module_name not in result:
                result[module_name] = []
            result[module_name].append(component)
        for module_name, module_components in result.items():
            result[module_name] = sorted(
                module_components,
                key=lambda x: (isinstance(self.nodes[x], CallableNode), x),
            )
        result = OrderedDict(sorted(result.items(), key=lambda x: x[0]))
        return result
