from ..common_imports import *
from .utils import DepKey, hash_dict
from .model import (
    Node,
    CallableNode,
    GlobalVarNode,
    TerminalNode,
)


from .shallow_versions import DAG


class Version:
    """
    Model of a "deep" version of a component that includes versions of its
    dependencies.
    """

    def __init__(
        self,
        component: DepKey,
        dynamic_deps_commits: Dict[DepKey, str],
        memoized_deps_content_versions: Dict[DepKey, Set[str]],
    ):
        ### raw data from the trace
        # the component whose dependencies are traced
        self.component = component
        # the content hashes of the direct dependencies
        self.direct_deps_commits = dynamic_deps_commits
        # pointers to content hashes of versions of memoized calls
        self.memoized_deps_content_versions = memoized_deps_content_versions

        ### cached data. These are set against a dependency state
        self._is_synced = False
        # the expanded set of dependencies, including all transitive
        # dependencies. Note this is a set of *content* hashes per dependency
        self._content_expansion: Dict[DepKey, Set[str]] = None
        # a hash uniquely identifying the content of dependencies of this version
        self._content_version: str = None
        # the semantic hashes of all dependencies for this version;
        # the system enforces that the semantic hash of a dependency is the same
        # for all commits of a component referenced by this version
        self._semantic_expansion: Dict[DepKey, str] = None
        # overall semantic hash of this version
        self._semantic_version: str = None

    @property
    def presentation(self) -> str:
        return f'Version of "{self.component[1]}" from module "{self.component[0]}" (content: {self.content_version}, semantic: {self.semantic_version})'

    @staticmethod
    def from_trace(
        component: DepKey, nodes: Dict[DepKey, Node], strict: bool = True
    ) -> "Version":
        dynamic_deps_commits = {}
        memoized_deps_content_versions = defaultdict(set)
        for dep_key, node in nodes.items():
            if isinstance(node, (CallableNode, GlobalVarNode)):
                dynamic_deps_commits[dep_key] = node.content_hash
            elif isinstance(node, TerminalNode):
                terminal_data = node.representation
                pointer_dep_key = terminal_data.dep_key
                version_content_hash = terminal_data.call_content_version
                memoized_deps_content_versions[pointer_dep_key].add(
                    version_content_hash
                )
            else:
                raise ValueError(f"Unexpected node type {type(node)}")
        return Version(
            component=component,
            dynamic_deps_commits=dynamic_deps_commits,
            memoized_deps_content_versions=dict(memoized_deps_content_versions),
        )

    ############################################################################
    ### methods for setting cached data from a versioning state
    ############################################################################
    def _set_content_expansion(self, all_versions: Dict[DepKey, Dict[str, "Version"]]):
        result = defaultdict(set)
        for dep_key, content_hash in self.direct_deps_commits.items():
            result[dep_key].add(content_hash)
        for (
            dep_key,
            memoized_content_versions,
        ) in self.memoized_deps_content_versions.items():
            for memoized_content_version in memoized_content_versions:
                referenced_version = all_versions[dep_key][memoized_content_version]
                for (
                    referenced_dep_key,
                    referenced_content_hashes,
                ) in referenced_version.content_expansion.items():
                    result[referenced_dep_key].update(referenced_content_hashes)
        self._content_expansion = dict(result)

    def _set_content_version(self):
        self._content_version = hash_dict(
            {
                dep_key: tuple(sorted(self.content_expansion[dep_key]))
                for dep_key in self.content_expansion
            }
        )

    def _set_semantic_expansion(
        self,
        component_dags: Dict[DepKey, DAG],
        all_versions: Dict[DepKey, Dict[str, "Version"]],
    ):
        result = {}
        # from own deps
        for dep_key, dep_content_hash in self.direct_deps_commits.items():
            dag = component_dags[dep_key]
            semantic_hash = dag.commits[dep_content_hash].semantic_hash
            result[dep_key] = semantic_hash
        # from pointers
        for (
            dep_key,
            memoized_content_versions,
        ) in self.memoized_deps_content_versions.items():
            for memoized_content_version in memoized_content_versions:
                dep_version_semantic_hashes = all_versions[dep_key][
                    memoized_content_version
                ].semantic_expansion
                overlap = set(result.keys()).intersection(
                    dep_version_semantic_hashes.keys()
                )
                if any(result[k] != dep_version_semantic_hashes[k] for k in overlap):
                    raise ValueError(
                        f"Version {self} has conflicting semantic hashes for {overlap}"
                    )
                result.update(dep_version_semantic_hashes)
        self._semantic_expansion = result
        self._semantic_version = hash_dict(result)

    def sync(
        self,
        component_dags: Dict[DepKey, DAG],
        all_versions: Dict[DepKey, Dict[str, "Version"]],
    ):
        """
        Set all the cached things in the correct order
        """
        self._set_content_expansion(all_versions=all_versions)
        self._set_content_version()
        self._set_semantic_expansion(
            component_dags=component_dags, all_versions=all_versions
        )
        self.set_synced()

    @property
    def content_version(self) -> str:
        assert self._content_version is not None
        return self._content_version

    @property
    def semantic_version(self) -> str:
        assert self._semantic_version is not None
        return self._semantic_version

    @property
    def semantic_expansion(self) -> Dict[DepKey, str]:
        assert self._semantic_expansion is not None
        return self._semantic_expansion

    @property
    def content_expansion(self) -> Dict[DepKey, Set[str]]:
        assert self._content_expansion is not None
        return self._content_expansion

    @property
    def support(self) -> Iterable[DepKey]:
        return self.content_expansion.keys()

    @property
    def is_synced(self) -> bool:
        return self._is_synced

    def set_synced(self):
        # it can only go from unsynced to synced
        if self._is_synced:
            raise ValueError("Version is already synced")
        self._is_synced = True

    def __repr__(self) -> str:
        return f"""
Version(
    dependencies={['.'.join(elt) for elt in self.support]},
)"""
