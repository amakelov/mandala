from ..common_imports import *
from .storage import Storage
from ..core.model import Ref, Call
from ..core.builtins_ import StructRef
from ..core.config import Provenance, parse_output_idx
from ..queries.weaver import ValNode, CallNode, StructOrientations, traverse_all
from ..queries.viz import GraphPrinter

BUILTIN_OP_IDS = ("__list___0", "__dict___0", "__set___0")
OP_ID_TO_STRUCT_NAME = {
    "__list___0": "lst",
    "__dict___0": "dct",
    "__set___0": "st",
}

OP_ID_TO_ELT_NAME = {
    "__list___0": "elt",
    "__dict___0": "val",
    "__set___0": "elt",
}

ELT_NAMES = ("elt", "val")
STRUCT_NAMES = ("lst", "dct", "st")
IDX_NAMES = ("idx", "key")


class ProvHelpers:
    """
    Tools for producing provenance graphs identical to runtime provenance graphs
    """

    def __init__(self, storage: Storage, prov_df: pd.DataFrame):
        self.storage = storage
        self.prov = prov_df.sort_index()  # improve performance of .loc

    def get_call_df(self, call_causal_uid: str) -> pd.DataFrame:
        """
        Get the provenance dataframe for the given call causal uid
        """
        #! very inefficient
        return self.prov[self.prov[Provenance.call_causal_uid] == call_causal_uid]

    def get_df_as_input(self, causal_uid: str) -> pd.DataFrame:
        """
        Get the provenance dataframe for the given causal uid as an input
        """
        key = (causal_uid, "input")
        if key not in self.prov.index:
            # return empty like self.prov
            return self.prov.loc[[]]
        else:
            return self.prov.loc[[key]]  # passing a list to .loc guarantees a dataframe

    def get_df_as_output(self, causal_uid: str) -> pd.DataFrame:
        """
        Get the provenance dataframe for the given causal uid as an output
        """
        key = (causal_uid, "output")
        if key not in self.prov.index:
            # return empty like self.prov
            return self.prov.loc[[]]
        else:
            res = self.prov.loc[[key]]  # passing a list to .loc guarantees a dataframe
            if res.shape[0] > 1:
                raise NotImplementedError(
                    f"causal uid {causal_uid} is the output of multiple calls"
                )
            return res

    def is_op_output(self, causal_uid: str) -> bool:
        """
        Check if the given causal uid is the output of a (non-struct) op call
        and verify that there is a unique such call
        """
        n = self.get_df_as_output(causal_uid=causal_uid).shape[0]
        if n > 1:
            raise NotImplementedError(
                f"causal uid {causal_uid} is the output of multiple op calls"
            )
        return n == 1

    def get_containing_structs(self, causal_uid: str) -> Tuple[List[str], List[str]]:
        """
        Return two lists: the causal UIDs of structs containing this ref, and
        the causal call UIDs of the corresponding structural calls
        """
        elt_rows = self.get_df_as_input(causal_uid=causal_uid)
        # restrict to structs containing this ref only
        elt_rows = elt_rows[elt_rows[Provenance.op_id].isin(BUILTIN_OP_IDS)]
        elt_rows = elt_rows[elt_rows[Provenance.name].isin(ELT_NAMES)]
        # get the call causal UIDs
        call_causal_uids = elt_rows[Provenance.call_causal_uid].tolist()
        # restrict to the rows containing the structs themselves
        x = self.prov[self.prov[Provenance.call_causal_uid].isin(call_causal_uids)]
        x = x[x[Provenance.name].isin(STRUCT_NAMES)]
        return (
            x.index.get_level_values(0).tolist(),
            x[Provenance.call_causal_uid].tolist(),
        )

    def get_struct_call_uids(self, causal_uid: str) -> List[str]:
        key = (causal_uid, "input")
        df = self.prov.loc[[key]]
        df = df[df[Provenance.op_id].isin(BUILTIN_OP_IDS)]
        df = df[df[Provenance.name].isin(STRUCT_NAMES)]
        return df[Provenance.call_causal_uid].tolist()

    def get_creator_chain(self, causal_uid: str) -> List[str]:
        """
        Return a list of call UIDs for the chain of calls creating a value (if any)
        """
        as_output_df = self.get_df_as_output(causal_uid=causal_uid)
        if as_output_df.shape[0] > 0:
            return as_output_df[Provenance.call_causal_uid].tolist()
        else:
            # gotta check for containing structs
            struct_uids, struct_call_uids = self.get_containing_structs(
                causal_uid=causal_uid
            )
            for struct_uid, struct_call_uid in zip(struct_uids, struct_call_uids):
                recursive_result = self.get_creator_chain(causal_uid=struct_uid)
                if len(recursive_result) > 0:
                    return [struct_call_uid] + self.get_creator_chain(
                        causal_uid=struct_uid
                    )
            return []

    def link_call_into_graph(
        self,
        call_causal_uid: str,
        refs: Dict[str, Ref],
        orientation: Optional[str],
        calls: Dict[str, Call],
    ) -> Call:
        """
        Load the call object from storage, link in into the current graph
        (creating `Ref` instances when they are not already in `refs`), and add
        pointwise constraints to each ref (via the full UID).

        ! Note that for structs we don't unify the indices of the struct with the
          rest of the graph to avoid unnatural clashes between refs. We also
          don't put the indices in the `refs` object.
        """
        # load the call
        call = self.storage.rel_adapter.call_get_lazy(
            uid=call_causal_uid, by_causal=True
        )
        if call.full_uid in calls:
            return calls[call.full_uid]
        # look up inputs/outputs in `refs` or add them there
        for name, inp in call.inputs.items():
            if call.func_op.is_builtin and name in IDX_NAMES:
                continue
            if inp.causal_uid in refs:
                call.inputs[name] = refs[inp.causal_uid]
            else:
                refs[inp.causal_uid] = inp
        for idx, outp in enumerate(call.outputs):
            if outp.causal_uid in refs:
                call.outputs[idx] = refs[outp.causal_uid]
            else:
                refs[outp.causal_uid] = outp
        # actual linking step
        call.link(orientation=orientation)
        # add pointwise constraints
        for ref in itertools.chain(call.inputs.values(), call.outputs):
            ref.query.constraint = [ref.full_uid]
        calls[call.full_uid] = call
        return call

    def step(self, ref: Ref, refs: Dict[str, Ref], calls: Dict[str, Call]) -> List[Ref]:
        """
        Given the (*non-index*) refs constructed so far and a ref in the graph,
        link the chain of calls creating this ref. If some of the refs along the
        way already exist, use those instead of creating new refs.
        """
        # build one step of the provenance graph
        creator_chain = self.get_creator_chain(causal_uid=ref.causal_uid)
        if not creator_chain:
            if isinstance(ref, StructRef):
                constituent_call_uids = self.get_struct_call_uids(
                    causal_uid=ref.causal_uid
                )
                elts = []
                for constituent_call_uid in constituent_call_uids:
                    call = self.link_call_into_graph(
                        call_causal_uid=constituent_call_uid,
                        refs=refs,
                        orientation=StructOrientations.construct,
                        calls=calls,
                    )
                    op_id = call.func_op.sig.versioned_internal_name
                    elts.append(call.inputs[OP_ID_TO_ELT_NAME[op_id]])
                return elts
            else:
                return []
        elif len(creator_chain) == 1:
            op_call = self.link_call_into_graph(
                call_causal_uid=creator_chain[0],
                refs=refs,
                orientation=None,
                calls=calls,
            )
            return list(op_call.inputs.values())
        else:
            for struct_call_uid in creator_chain[:-1]:
                self.link_call_into_graph(
                    call_causal_uid=struct_call_uid,
                    refs=refs,
                    orientation=StructOrientations.destruct,
                    calls=calls,
                )
            op_call = self.link_call_into_graph(
                call_causal_uid=creator_chain[-1],
                refs=refs,
                orientation=None,
                calls=calls,
            )
            return list(op_call.inputs.values())

    def get_graph(self, full_uid: str) -> Tuple[Set[ValNode], Set[CallNode]]:
        """
        Given the full UID of a ref, recover the full provenance graph for that
        ref
        """
        refs = {}
        calls = {}
        res = Ref.from_full_uid(full_uid=full_uid)
        refs[res.causal_uid] = res
        queue = [res]
        while queue:
            ref = queue.pop()
            queue.extend(self.step(ref=ref, refs=refs, calls=calls))
        return traverse_all(vqs=[res.query], direction="backward")
