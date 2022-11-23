import collections
import logging
import os

from .python import Python
from .model import GROUP_TYPE, OWNER_CONST, Edge, Group, Node, Variable, flatten


class LanguageParams:
    """
    Shallow structure to make storing language-specific parameters cleaner
    """

    def __init__(self, source_type="script", ruby_version="27"):
        self.source_type = source_type
        self.ruby_version = ruby_version


class SubsetParams:
    """
    Shallow structure to make storing subset-specific parameters cleaner.
    """

    def __init__(self, target_function, upstream_depth, downstream_depth):
        self.target_function = target_function
        self.upstream_depth = upstream_depth
        self.downstream_depth = downstream_depth


def get_sources_and_language(raw_source_paths, language):
    """
    Given a list of files and directories, return just files.
    If we are not passed a language, determine it.
    Filter out files that are not of that language

    :param list[str] raw_source_paths: file or directory paths
    :param str|None language: Input language
    :rtype: (list, str)
    """

    individual_files = []
    for source in sorted(raw_source_paths):
        if os.path.isfile(source):
            individual_files.append((source, True))
            continue
        for root, _, files in os.walk(source):
            for f in files:
                individual_files.append((os.path.join(root, f), False))

    if not individual_files:
        raise AssertionError("No source files found from %r" % raw_source_paths)
    logging.info("Found %d files from sources argument.", len(individual_files))

    if not language:
        language = "py"

    sources = set()
    for source, explicity_added in individual_files:
        if explicity_added or source.endswith("." + language):
            sources.add(source)
        else:
            logging.info(
                "Skipping %r which is not a %s file. "
                "If this is incorrect, include it explicitly.",
                source,
                language,
            )

    if not sources:
        raise AssertionError(
            "Could not find any source files given {raw_source_paths} "
            "and language {language}."
        )

    sources = sorted(list(sources))
    logging.info("Processing %d source file(s)." % (len(sources)))
    for source in sources:
        logging.info("  " + source)

    return sources, language


def make_file_group(tree, filename, extension):
    """
    Given an AST for the entire file, generate a file group complete with
    subgroups, nodes, etc.

    :param tree ast:
    :param filename str:
    :param extension str:

    :rtype: Group
    """
    language = Python

    subgroup_trees, node_trees, body_trees = language.separate_namespaces(tree)
    group_type = GROUP_TYPE.FILE
    token = os.path.split(filename)[-1].rsplit("." + extension, 1)[0]
    line_number = 0
    display_name = "File"
    import_tokens = language.file_import_tokens(filename)

    file_group = Group(
        token, group_type, display_name, import_tokens, line_number, parent=None
    )
    for node_tree in node_trees:
        for new_node in language.make_nodes(node_tree, parent=file_group):
            file_group.add_node(new_node)

    file_group.add_node(
        language.make_root_node(body_trees, parent=file_group), is_root=True
    )

    for subgroup_tree in subgroup_trees:
        file_group.add_subgroup(
            language.make_class_group(subgroup_tree, parent=file_group)
        )
    return file_group


def _find_link_for_call(call, node_a, all_nodes):
    """
    Given a call that happened on a node (node_a), return the node
    that the call links to and the call itself if >1 node matched.

    :param call Call:
    :param node_a Node:
    :param all_nodes list[Node]:

    :returns: The node it links to and the call if >1 node matched.
    :rtype: (Node|None, Call|None)
    """

    all_vars = node_a.get_variables(call.line_number)

    for var in all_vars:
        var_match = call.matches_variable(var)
        if var_match:
            # Unknown modules (e.g. third party) we don't want to match)
            if var_match == OWNER_CONST.UNKNOWN_MODULE:
                return None, None
            assert isinstance(var_match, Node)
            return var_match, None

    possible_nodes = []
    if call.is_attr():
        for node in all_nodes:
            # checking node.parent != node_a.file_group() prevents self linkage in cases like
            # function a() {b = Obj(); b.a()}
            if call.token == node.token and node.parent != node_a.file_group():
                possible_nodes.append(node)
    else:
        for node in all_nodes:
            if (
                call.token == node.token
                and isinstance(node.parent, Group)
                and node.parent.group_type == GROUP_TYPE.FILE
            ):
                possible_nodes.append(node)
            elif call.token == node.parent.token and node.is_constructor:
                possible_nodes.append(node)

    if len(possible_nodes) == 1:
        return possible_nodes[0], None
    if len(possible_nodes) > 1:
        return None, call
    return None, None


def _find_links(node_a, all_nodes):
    """
    Iterate through the calls on node_a to find everything the node links to.
    This will return a list of tuples of nodes and calls that were ambiguous.

    :param Node node_a:
    :param list[Node] all_nodes:
    :param BaseLanguage language:
    :rtype: list[(Node, Call)]
    """

    links = []
    for call in node_a.calls:
        lfc = _find_link_for_call(call, node_a, all_nodes)
        assert not isinstance(lfc, Group)
        links.append(lfc)
    return list(filter(None, links))


def map_it(
    sources,
    extension,
    no_trimming=False,
    exclude_namespaces=None,
    exclude_functions=None,
    include_only_namespaces=None,
    include_only_functions=None,
    skip_parse_errors=False,
    lang_params=None,
):
    """
    Given a language implementation and a list of filenames, do these things:
    1. Read/parse source ASTs
    2. Find all groups (classes/modules) and nodes (functions) (a lot happens here)
    3. Trim namespaces / functions that we don't want
    4. Consolidate groups / nodes given all we know so far
    5. Attempt to resolve the variables (point them to a node or group)
    6. Find all calls between all nodes
    7. Loudly complain about duplicate edges that were skipped
    8. Trim nodes that didn't connect to anything

    :param list[str] sources:
    :param str extension:
    :param bool no_trimming:
    :param list exclude_namespaces:
    :param list exclude_functions:
    :param list include_only_namespaces:
    :param list include_only_functions:
    :param bool skip_parse_errors:
    :param LanguageParams lang_params:

    :rtype: (list[Group], list[Node], list[Edge])
    """

    language = Python

    # 0. Assert dependencies
    language.assert_dependencies()

    # 1. Read/parse source ASTs
    file_ast_trees = []
    for source in sources:
        try:
            file_ast_trees.append((source, language.get_tree(source, lang_params)))
        except Exception as ex:
            if skip_parse_errors:
                logging.warning("Could not parse %r. (%r) Skipping...", source, ex)
            else:
                raise ex

    # 2. Find all groups (classes/modules) and nodes (functions) (a lot happens here)
    file_groups = []
    for source, file_ast_tree in file_ast_trees:
        file_group = make_file_group(file_ast_tree, source, extension)
        file_groups.append(file_group)

    # 3. Trim namespaces / functions to exactly what we want
    if exclude_namespaces or include_only_namespaces:
        file_groups = _limit_namespaces(
            file_groups, exclude_namespaces, include_only_namespaces
        )
    if exclude_functions or include_only_functions:
        file_groups = _limit_functions(
            file_groups, exclude_functions, include_only_functions
        )

    # 4. Consolidate structures
    all_subgroups = flatten(g.all_groups() for g in file_groups)
    all_nodes = flatten(g.all_nodes() for g in file_groups)

    nodes_by_subgroup_token = collections.defaultdict(list)
    for subgroup in all_subgroups:
        if subgroup.token in nodes_by_subgroup_token:
            logging.warning(
                "Duplicate group name %r. Naming collision possible.", subgroup.token
            )
        nodes_by_subgroup_token[subgroup.token] += subgroup.nodes

    for group in file_groups:
        for subgroup in group.all_groups():
            subgroup.inherits = [
                nodes_by_subgroup_token.get(g) for g in subgroup.inherits
            ]
            subgroup.inherits = list(filter(None, subgroup.inherits))
            for inherit_nodes in subgroup.inherits:
                for node in subgroup.nodes:
                    node.variables += [
                        Variable(n.token, n, n.line_number) for n in inherit_nodes
                    ]

    # 5. Attempt to resolve the variables (point them to a node or group)
    for node in all_nodes:
        node.resolve_variables(file_groups)

    # Not a step. Just log what we know so far
    logging.info("Found groups %r." % [g.label() for g in all_subgroups])
    logging.info(
        "Found nodes %r." % sorted(n.token_with_ownership() for n in all_nodes)
    )
    logging.info(
        "Found calls %r."
        % sorted(list(set(c.to_string() for c in flatten(n.calls for n in all_nodes))))
    )
    logging.info(
        "Found variables %r."
        % sorted(
            list(set(v.to_string() for v in flatten(n.variables for n in all_nodes)))
        )
    )

    # 6. Find all calls between all nodes
    bad_calls = []
    edges = []
    for node_a in list(all_nodes):
        links = _find_links(node_a, all_nodes)
        for node_b, bad_call in links:
            if bad_call:
                bad_calls.append(bad_call)
            if not node_b:
                continue
            edges.append(Edge(node_a, node_b))

    # 7. Loudly complain about duplicate edges that were skipped
    bad_calls_strings = set()
    for bad_call in bad_calls:
        bad_calls_strings.add(bad_call.to_string())
    bad_calls_strings = list(sorted(list(bad_calls_strings)))
    if bad_calls_strings:
        logging.info(
            "Skipped processing these calls because the algorithm "
            "linked them to multiple function definitions: %r." % bad_calls_strings
        )

    if no_trimming:
        return file_groups, all_nodes, edges

    # 8. Trim nodes that didn't connect to anything
    nodes_with_edges = set()
    for edge in edges:
        nodes_with_edges.add(edge.node0)
        nodes_with_edges.add(edge.node1)

    for node in all_nodes:
        if node not in nodes_with_edges:
            node.remove_from_parent()

    for file_group in file_groups:
        for group in file_group.all_groups():
            if not group.all_nodes():
                group.remove_from_parent()

    file_groups = [g for g in file_groups if g.all_nodes()]
    all_nodes = list(nodes_with_edges)

    if not all_nodes:
        logging.warning(
            "No functions found! Most likely, your file(s) do not have "
            "functions that call each other. Note that to generate a flowchart, "
            "you need to have both the function calls and the function "
            "definitions. Or, you might be excluding too many "
            "with --exclude-* / --include-* / --target-function arguments. "
        )
        logging.warning("Code2flow will generate an empty output file.")

    return file_groups, all_nodes, edges


def _limit_namespaces(file_groups, exclude_namespaces, include_only_namespaces):
    """
    Exclude namespaces (classes/modules) which match any of the exclude_namespaces

    :param list[Group] file_groups:
    :param list exclude_namespaces:
    :param list include_only_namespaces:
    :rtype: list[Group]
    """

    removed_namespaces = set()

    for group in list(file_groups):
        if group.token in exclude_namespaces:
            for node in group.all_nodes():
                node.remove_from_parent()
            removed_namespaces.add(group.token)
        if include_only_namespaces and group.token not in include_only_namespaces:
            for node in group.nodes:
                node.remove_from_parent()
            removed_namespaces.add(group.token)

        for subgroup in group.all_groups():
            print(subgroup, subgroup.all_parents())
            if subgroup.token in exclude_namespaces:
                for node in subgroup.all_nodes():
                    node.remove_from_parent()
                removed_namespaces.add(subgroup.token)
            if (
                include_only_namespaces
                and subgroup.token not in include_only_namespaces
                and all(
                    p.token not in include_only_namespaces
                    for p in subgroup.all_parents()
                )
            ):
                for node in subgroup.nodes:
                    node.remove_from_parent()
                removed_namespaces.add(group.token)

    for namespace in exclude_namespaces:
        if namespace not in removed_namespaces:
            logging.warning(
                f"Could not exclude namespace '{namespace}' "
                "because it was not found."
            )
    return file_groups


def _limit_functions(file_groups, exclude_functions, include_only_functions):
    """
    Exclude nodes (functions) which match any of the exclude_functions

    :param list[Group] file_groups:
    :param list exclude_functions:
    :param list include_only_functions:
    :rtype: list[Group]
    """

    removed_functions = set()

    for group in list(file_groups):
        for node in group.all_nodes():
            if node.token in exclude_functions or (
                include_only_functions and node.token not in include_only_functions
            ):
                node.remove_from_parent()
                removed_functions.add(node.token)

    for function_name in exclude_functions:
        if function_name not in removed_functions:
            logging.warning(
                f"Could not exclude function '{function_name}' "
                "because it was not found."
            )
    return file_groups
