from typing import Literal
import textwrap
from ..common_imports import *
from ..utils import get_content_hash
from ..config import Config
from ..utils import ask_user
from ..viz import _get_colorized_diff, _get_diff

if Config.has_rich:
    from rich.tree import Tree
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax


# TODO: figure out how to apply compact diffs
def get_diff(a: str, b: str) -> Tuple[str, str]:
    """
    Get a diff between two strings
    """
    return (a, b)


def apply_diff(b: str, diff: Tuple[str, str]) -> str:
    """
    Apply a diff to a string
    """
    return diff[0]


class Commit:
    """
    Tracks versions of the "shallow" content of a single component. For
    functions/methods, this is just the function source code, without any
    dependencies. For global variables, this is just (the content hash of) the
    value of the variable.
    """

    def __init__(
        self,
        parents: List[str],
        diffs: List[Any],
        content_hash: str,
        semantic_hash: str,
        content: Optional[str],
    ):
        # content hashes of parent commits
        self.parents = parents  # currently there may be at most one parent
        # diffs between this commit and its parents
        self.diffs = diffs
        self.content_hash = content_hash
        # content hash of the semantic version this commit is associated with
        self.semantic_hash = semantic_hash
        # content of this commit, if it is a root commit
        self._content = content
        self.check_invariants()

    def check_invariants(self):
        assert len(self.parents) == len(self.diffs)
        assert len(self.parents) > 0 or self._content is not None

    def __repr__(self) -> str:
        return f"Commit(content_hash={self.content_hash}, semantic_hash={self.semantic_hash}, parents={self.parents})"


T = TypeVar("T")
from typing import Generic


class ContentAdapter(Generic[T]):
    def get_diff(self, a: T, b: T) -> Any:
        """
        Get a diff between two objects
        """
        raise NotImplementedError()

    def apply_diff(self, b: T, diff: Any) -> T:
        """
        Apply a diff to an object
        """
        raise NotImplementedError()

    def get_presentable_content(self, content: T) -> str:
        """
        Get a presentable string representation of the content
        """
        raise NotImplementedError()

    def get_content_hash(self, content: T) -> str:
        raise NotImplementedError()


class StringContentAdapter(ContentAdapter[str]):
    def get_diff(self, a: str, b: str) -> Tuple[str, str]:
        """
        Get a diff between two strings
        """
        return get_diff(a, b)

    def apply_diff(self, b: str, diff: Tuple[str, str]) -> str:
        """
        Apply a diff to a string
        """
        return apply_diff(b, diff)

    def get_presentable_content(self, content: str) -> str:
        """
        Get a presentable string representation of the content
        """
        return content

    def get_content_hash(self, content: str) -> str:
        return get_content_hash(content)


GVContent = Tuple[str, str]  # (content hash, repr)


class GlobalVariableContentAdapter(ContentAdapter[GVContent]):
    def get_diff(self, a: GVContent, b: GVContent) -> Tuple[GVContent, GVContent]:
        """
        Get a diff between two global variable contents
        """
        return (a, b)

    def apply_diff(self, b: GVContent, diff: Tuple[GVContent, GVContent]) -> GVContent:
        """
        Apply a diff to a global variable content
        """
        return diff[0]

    def get_presentable_content(self, content: GVContent) -> str:
        return content[1]

    def get_content_hash(self, content: GVContent) -> str:
        return content[0]


class DAG(Generic[T]):
    """
    Organizes the shallow versions of a single component in a `git`-like DAG.
    """

    def __init__(self, content_type: Literal["code", "global_variable"] = "code"):
        # content hash of the current head
        self.head: Optional[str] = None
        self.commits: Dict[str, Commit] = {}
        self._initial_commit: Optional[T] = None
        if content_type == "code":
            self.content_adapter = StringContentAdapter()
        elif content_type == "global_variable":
            self.content_adapter = GlobalVariableContentAdapter()
        else:
            raise ValueError(f"Invalid content_type: {content_type}")
        self.check_invariants()

    @property
    def initial_commit(self) -> str:
        assert self.head is not None
        return self._initial_commit

    def check_invariants(self):
        if self.head is not None:
            assert self.head in self.commits
        for commit, commit_obj in self.commits.items():
            commit_obj.check_invariants()
            assert all(p in self.commits for p in commit_obj.parents)
            assert commit_obj.content_hash == commit
            assert commit_obj.semantic_hash in self.commits

    def get_current_content(self) -> T:
        # return the full content of the current head
        assert self.head is not None
        return self.get_content(commit=self.head)

    def get_presentable_content(self, commit: str) -> str:
        return self.content_adapter.get_presentable_content(
            content=self.get_content(commit=commit)
        )

    def get_content(self, commit: str) -> T:
        # return the full content of a commit given its content hash
        if commit not in self.commits:
            raise ValueError(f"Commit {commit} not in DAG")
        commit_obj = self.commits[commit]
        if commit_obj._content is not None:
            return commit_obj._content
        else:
            parent_content = self.get_content(commit_obj.parents[0])
            return self.content_adapter.apply_diff(parent_content, commit_obj.diffs[0])

    def init(self, initial_content: T) -> str:
        """
        Initialize the DAG with the initial content, and set the head to the
        initial commit. Return the content hash of the initial commit.
        """
        # initialize the DAG with the initial content
        assert self.head is None
        content_hash = self.content_adapter.get_content_hash(content=initial_content)
        semantic_hash = content_hash
        commit = Commit(
            parents=[],
            diffs=[],
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            content=initial_content,
        )
        self.head = content_hash
        self.commits[content_hash] = commit
        self._initial_commit = content_hash
        return content_hash

    def checkout(self, commit: str, implicit_merge: bool = False):
        """
        checkout an *existing* commit, i.e. set the head to the given commit.
        if implicit_merge is True, an edge will be added from the new head to
        the old head
        """
        if commit not in self.commits:
            raise ValueError(f"Commit {commit} not in DAG")
        if implicit_merge:
            raise NotImplementedError
        self.head = commit

    def commit(self, content: T, is_semantic_change: Optional[bool] = None) -> str:
        """
        Commit a *new* version of the content and return the content hash
        """
        assert self.head is not None
        content_hash = self.content_adapter.get_content_hash(content=content)
        assert content_hash not in self.commits
        head_commit = self.commits[self.head]
        if is_semantic_change is None:
            presentable_diff = get_diff(
                self.content_adapter.get_presentable_content(content),
                self.get_presentable_content(commit=self.head),
            )
            if Config.has_rich:
                colorized_diff = _get_colorized_diff(
                        current=presentable_diff[1], new=presentable_diff[0],
                        colorize=False,
                    )
                panel = Panel(
                    Syntax(
                        colorized_diff,
                        lexer="diff",
                        line_numbers=True,
                        theme="solarized-light",
                    ),
                    title="Diff",
                )
                rich.print(panel)
            else:
                colorized_diff = _get_colorized_diff(
                    current=presentable_diff[1], new=presentable_diff[0],
                    colorize=True,
                )
                print(colorized_diff)
            answer = ask_user(
                question="Does this change require recomputation of dependent calls?\nWARNING: if the change created new dependencies and you choose 'no', you should add them by hand or risk missing changes in them.\nAnswer: [y]es/[n]o/[a]bort",
                valid_options=["y", "n", "a"],
            )
            print(f'You answered: "{answer}"')
            if answer == "a":
                raise ValueError("Aborting commit")
            is_semantic_change = answer == "y"
        if is_semantic_change:
            semantic_hash = content_hash
        else:
            semantic_hash = head_commit.semantic_hash
        diff = self.content_adapter.get_diff(
            content, self.get_content(commit=self.head)
        )
        commit = Commit(
            parents=[self.head],
            diffs=[diff],
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            content=None,
        )
        self.head = content_hash
        self.commits[content_hash] = commit
        return content_hash

    def sync(self, content: T, is_semantic_change: Optional[bool] = None) -> str:
        """
        if the content:
          - is the current head, do nothing
          - is in the DAG, checkout the content
          - is not in the DAG, commit the content

        return the content hash
        """
        assert self.head is not None
        content_hash = self.content_adapter.get_content_hash(content)
        if self.head == content_hash:
            result = self.head
        elif content_hash in self.commits:
            self.checkout(content_hash)
            result = self.head
        else:
            result = self.commit(content, is_semantic_change=is_semantic_change)
        assert self.head == content_hash
        return result

    ############################################################################
    ### visualization and printing
    ############################################################################
    def _get_tree_neighbors_representation(self) -> Dict[str, Set[str]]:
        """
        Get a {parent: {children}} representation of the tree underlying the DAG
        (obtained by following the first parent of each commit).
        """
        result = defaultdict(set)
        for commit in self.commits.values():
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                result[parent].add(commit.content_hash)
        return dict(result)

    def get_commit_presentation(
        self, commit: str, diff_only: bool, include_metadata: bool = False
    ) -> Tuple[str, str]:
        if diff_only:
            if commit == self.initial_commit:
                content_to_show = self.get_presentable_content(commit)
                content_type = "code"
            else:
                parent_content = self.get_content(self.commits[commit].parents[0])
                child_content = self.content_adapter.apply_diff(
                    b=parent_content, diff=self.commits[commit].diffs[0]
                )
                parent_presentable_content = (
                    self.content_adapter.get_presentable_content(content=parent_content)
                )
                child_presentable_content = (
                    self.content_adapter.get_presentable_content(content=child_content)
                )
                content_to_show = _get_diff(
                    current=parent_presentable_content,
                    new=child_presentable_content,
                )
                content_type = "diff"
        else:
            content_to_show = self.get_presentable_content(commit)
            content_type = "code"

        content_version = commit
        semantic_version = self.commits[commit].semantic_hash

        header_lines = []
        if commit == self.head:
            header_lines.append(f"### ===HEAD===")
        if include_metadata:
            header_lines.append(f"### content_commit={content_version}")
            header_lines.append(f"### semantic_commit={semantic_version}")
        if len(header_lines) > 0:
            header = "\n".join(header_lines)
            content_to_show = f"{header}\n{content_to_show}"
        return content_to_show, content_type

    if Config.has_rich:
        from rich.panel import Panel
        from rich.tree import Tree

        def get_commit_content_rich(
            self,
            commit: str,
            diff_only: bool = False,
            title: Optional[str] = None,
            include_metadata: bool = False,
        ) -> Panel:
            """
            Get a rich panel representing the content and metadata of a commit.
            """
            content_to_show, content_type = self.get_commit_presentation(
                commit=commit, diff_only=diff_only, include_metadata=include_metadata
            )
            if title is not None:
                title = Text(title, style="bold")
            lexer = "python" if content_type == "code" else "diff"
            content = Syntax(
                content_to_show,
                lexer=lexer,
                line_numbers=False,
                theme="solarized-light",
            )
            return Panel(renderable=content, title=title)

        def get_tree_rich(
            self, compact: bool = False, include_metadata: bool = False
        ) -> Tree:
            """
            Get a rich tree representing the tree underlying the DAG (obtained
            by following the first parent of each commit).
            """
            assert Config.has_rich
            if self.head is None:
                return Tree(label="DAG(head=None)")
            tree_neighbors = self._get_tree_neighbors_representation()
            tree_objs = {}
            initial_commit = self.initial_commit

            result = Tree(
                label=self.get_commit_content_rich(
                    initial_commit, include_metadata=include_metadata
                )
            )
            tree_objs[initial_commit] = result

            def grow(commit: str):
                if commit in tree_neighbors:
                    for child in tree_neighbors[commit]:
                        current_tree = tree_objs[commit]
                        new_tree = current_tree.add(
                            self.get_commit_content_rich(
                                child,
                                diff_only=compact,
                                include_metadata=include_metadata,
                            )
                        )
                        tree_objs[child] = new_tree
                        grow(child)

            grow(initial_commit)
            return result

    def __repr__(self) -> str:
        num_content = len(self.commits)
        num_semantic = len(set(c.semantic_hash for c in self.commits.values()))
        return f"DAG(head={self.head}) with {num_content} content version(s) and {num_semantic} semantic version(s)"

    def show(
        self, compact: bool = False, plain: bool = False, include_metadata: bool = False
    ):
        if Config.has_rich and not plain:
            rich.print(
                self.get_tree_rich(compact=compact, include_metadata=include_metadata)
            )
            return
        else:
            if self.head is None:
                return "DAG(head=None)"
            commits = list(self.commits.keys())
            commits = [self.head] + [k for k in commits if k != self.head]
            lines = []
            lines.append(
                self.get_commit_presentation(
                    commit=self.initial_commit,
                    diff_only=compact,
                    include_metadata=include_metadata,
                )[0]
            )
            lines.append("--------")
            tree_neighbors = self._get_tree_neighbors_representation()

            def visit(commit: str, depth: int):
                if commit in tree_neighbors:
                    for child in tree_neighbors[commit]:
                        child_text = self.get_commit_presentation(
                            commit=child,
                            diff_only=compact,
                            include_metadata=include_metadata,
                        )[0]
                        child_text = textwrap.indent(child_text, "    " * (depth + 1))
                        lines.append(child_text)
                        lines.append("--------")
                        visit(child, depth + 1)

            visit(self.initial_commit, 0)
            print("\n".join(lines))

    @property
    def size(self) -> int:
        return len(self.commits)

    @property
    def semantic_size(self) -> int:
        return len(set(c.semantic_hash for c in self.commits.values()))
