from ..common_imports import *
from ..core.model import Call, Ref, collect_detached
from ..storages.rels import RelAdapter, VersionAdapter
from ..deps.versioner import Versioner
from ..storages.kv import KVCache, InMemoryStorage, MultiProcInMemoryStorage
from ..storages.rel_impls.utils import Transactable, transaction, Connection


class Cache(Transactable):
    """
    A layer between calls happening in the context and the persistent storage.
    Also responsible for detaching objects from the computational graph.
    """

    def __init__(
        self,
        rel_adapter: RelAdapter,
    ):
        self.rel_adapter = rel_adapter
        # uid -> detached call
        self.call_cache_by_uid = InMemoryStorage()
        # causal uid -> detached call
        self.call_cache_by_causal = InMemoryStorage()
        # uid -> unlinked ref without causal
        self.obj_cache = InMemoryStorage()

    def mcache_call_and_objs(self, calls: List[Call]) -> None:
        # a more efficient version of `cache_call_and_objs` for multiple calls
        # that avoids calling `unlinked` multiple times on the same object,
        # which could be expensive for large collections.
        unique_vrefs = {}
        unique_calls = {}
        for call in calls:
            for vref in itertools.chain(call.inputs.values(), call.outputs):
                unique_vrefs[vref.uid] = vref
            unique_calls[call.causal_uid] = call
        for vref in unique_vrefs.values():
            self.obj_cache[vref.uid] = vref.unlinked(keep_causal=False)
        for call in unique_calls.values():
            self.cache_call(causal_uid=call.causal_uid, call=call)

    def cache_call_and_objs(self, call: Call) -> None:
        for vref in itertools.chain(call.inputs.values(), call.outputs):
            self.obj_cache[vref.uid] = vref.unlinked(keep_causal=False)
        self.cache_call(causal_uid=call.causal_uid, call=call)

    def cache_call(self, causal_uid: str, call: Call) -> None:
        self.call_cache_by_causal.set(causal_uid, call.detached())
        self.call_cache_by_uid.set(call.uid, call.detached())

    def mattach(
        self, vrefs: List[Ref], shallow: bool = False, _attach_atoms: bool = True
    ):
        """
        Regardless of `shallow`, recursively find all the refs that can be found
        in the cache. Then pass what's left to the storage method.
        """
        vrefs = collect_detached(vrefs, include_transient=False)
        cur_frontier = vrefs
        new_frontier = []
        not_found = []
        while len(cur_frontier) > 0:
            for vref in cur_frontier:
                if (
                    vref.uid in self.obj_cache.keys()
                    and self.obj_cache[vref.uid].in_memory
                ):
                    vref.attach(reference=self.obj_cache[vref.uid])
                    new_frontier.extend(
                        collect_detached([vref], include_transient=False)
                    )
                else:
                    not_found.append(vref)
            cur_frontier = new_frontier
            new_frontier = []
        if len(not_found) > 0:
            self.rel_adapter.mattach(
                vrefs=not_found, shallow=shallow, _attach_atoms=_attach_atoms
            )

    def obj_get(self, obj_uid: str, causal_uid: Optional[str] = None) -> Ref:
        """
        Get the given object from the cache or the storage.
        """
        #! note that this is not transactional to avoid creating a connection
        #! when the object is already in the cache
        if self.obj_cache.exists(obj_uid):
            res = self.obj_cache.get(obj_uid)
        else:
            res = self.rel_adapter.obj_get(uid=obj_uid)
        res = res.clone()
        if causal_uid is not None:
            res.causal_uid = causal_uid
        return res

    def call_exists(self, uid: str, by_causal: bool) -> bool:
        #! note that this is not transactional to avoid creating a connection
        #! when the object is already in the cache
        if by_causal:
            return self.call_cache_by_causal.exists(
                uid
            ) or self.rel_adapter.call_exists(uid=uid, by_causal=True)
        else:
            return self.call_cache_by_uid.exists(uid) or self.rel_adapter.call_exists(
                uid=uid, by_causal=False
            )

    @transaction()
    def call_mget(
        self,
        uids: List[str],
        versioned_ui_name: str,
        by_causal: bool = True,
        lazy: bool = True,
        conn: Optional[Connection] = None,
    ) -> List[Call]:
        if not by_causal:
            raise NotImplementedError()
        if not lazy:
            raise NotImplementedError()
        res = [None for _ in uids]
        missing_indices = []
        missing_uids = []
        for i, uid in enumerate(uids):
            if self.call_cache_by_causal.exists(uid):
                res[i] = self.call_cache_by_causal.get(uid)
            else:
                missing_indices.append(i)
                missing_uids.append(uid)
        if len(missing_uids) > 0:
            lazy_calls = self.rel_adapter.mget_call_lazy(
                versioned_ui_name=versioned_ui_name,
                uids=missing_uids,
                by_causal=True,
                conn=conn,
            )
            for i, lazy_call in zip(missing_indices, lazy_calls):
                res[i] = lazy_call
        return res

    def call_get(self, uid: str, by_causal: bool, lazy: bool = True) -> Call:
        """
        Return a *detached* call with the given UID, if it exists.
        """
        #! note that this is not transactional to avoid creating a connection
        #! when the object is already in the cache
        if by_causal and self.call_cache_by_causal.exists(uid):
            return self.call_cache_by_causal.get(uid)
        elif not by_causal and self.call_cache_by_uid.exists(uid):
            return self.call_cache_by_uid.get(uid)
        else:
            lazy_call = self.rel_adapter.call_get_lazy(uid=uid, by_causal=by_causal)
            if not lazy:
                #! you need to be more careful here about guarantees provided by
                #! the cache
                raise NotImplementedError
                # # load the values of the inputs and outputs
                # inputs = {
                #     k: self.obj_get(v.uid)
                #     for k, v in lazy_call.inputs.items()
                # }
                # outputs = [self.obj_get(v.uid) for v in lazy_call.outputs]
                # call_without_outputs = lazy_call.set_input_values(inputs=inputs)
                # call = call_without_outputs.set_output_values(outputs=outputs)
                # return call
            else:
                return lazy_call

    @transaction()
    def commit(
        self,
        calls: Optional[List[Call]] = None,
        versioner: Optional[Versioner] = None,
        version_adapter: VersionAdapter = None,
        conn: Optional[Connection] = None,
    ):
        """
        Flush dirty (written since last time) calls and objs from the cache to
        persistent storage, and mark them as clean.
        """
        if calls is None:
            new_objs = {
                key: self.obj_cache.get(key) for key in self.obj_cache.dirty_entries
            }
            new_calls = [
                self.call_cache_by_causal.get(key)
                for key in self.call_cache_by_causal.dirty_entries
            ]
        else:
            #! if calls are provided, we assume they are attached
            new_objs = {}
            for call in calls:
                for vref in itertools.chain(call.inputs.values(), call.outputs):
                    new_obj = self.obj_cache[vref.uid]
                    assert new_obj.in_memory
                    new_objs[vref.uid] = new_obj
            new_calls = calls
        self.rel_adapter.obj_sets(new_objs, conn=conn)
        self.rel_adapter.upsert_calls(new_calls, conn=conn)
        # if self.evict_on_commit:
        #     self.evict_caches()
        if versioner is not None:
            version_adapter.dump_state(state=versioner, conn=conn)
        self.clear_all()

    @transaction()
    def preload_objs(self, uids: List[str], conn: Optional[Connection] = None):
        """
        Put the objects with the given UIDs in the cache. Should be used for
        bulk loading b/c it opens a connection
        """
        uids_not_in_cache = [uid for uid in uids if not self.obj_cache.exists(uid)]
        for uid, vref in zip(
            uids_not_in_cache,
            self.rel_adapter.obj_gets(uids=uids_not_in_cache, conn=conn),
        ):
            self.obj_cache.set(k=uid, v=vref)

    def evict_all(self):
        self.call_cache_by_causal.evict_all()
        self.call_cache_by_uid.evict_all()
        self.obj_cache.evict_all()

    def clear_all(self):
        self.call_cache_by_causal.clear_all()
        self.call_cache_by_uid.clear_all()
        self.obj_cache.clear_all()

    def detach_all(self):
        for k, v in self.call_cache_by_causal.items():
            self.call_cache_by_causal[k] = v.detached()
        for k, v in self.call_cache_by_uid.items():
            self.call_cache_by_uid[k] = v.detached()
        for k, v in self.obj_cache.items():
            self.obj_cache[k] = v.detached()

    def _get_connection(self) -> Connection:
        return self.rel_adapter._get_connection()

    def _end_transaction(self, conn: Connection):
        return self.rel_adapter._end_transaction(conn=conn)
