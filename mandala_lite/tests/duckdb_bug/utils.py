from ..test_stateful import *


def load_bug() -> Tuple[Storage, List[FuncOp]]:
    db_dump_path = Path(__file__).parent.absolute() / "db_dump/"
    storage = Storage()
    for table in storage.rel_storage.get_tables():
        storage.rel_storage.execute_no_results(query=f"DROP TABLE {table};")
    storage.rel_storage.execute_no_results(query=f"IMPORT DATABASE '{db_dump_path}';")
    path = Path(__file__).parent / f"bug.cloudpickle"
    with open(path, "rb") as f:
        data = cloudpickle.load(f)
    return storage, data


def debug():
    storage, ops = load_bug()
    ops_by_name = {op.sig.ui_name: op for op in ops}

    with storage.query() as q:
        var_0 = Q()
        # var_2 = Q()
        var_3 = Q()
        # var_1, = NQqEefRWij(fSIRzTHrHZ=var_0)
        # var_1, = storage.call_query(op=ops_by_name["NQqEefRWij"], inputs={"fSIRzTHrHZ": var_0})
        # TYQOaNFvpU(CffuGFgtJs=var_2, TemKopZjZI=var_3)
        # storage.call_query(op=ops_by_name["TYQOaNFvpU"], inputs={"CffuGFgtJs": var_2, "TemKopZjZI": var_3})
        # var_4, var_5, var_6 = RvEJgwBuNO(JECHqdZJaf=var_0)
        var_4, var_5, var_6 = storage.call_query(
            op=ops_by_name["RvEJgwBuNO"], inputs={"JECHqdZJaf": var_0}
        )
        # var_7, var_8, var_9 = RvEJgwBuNO(JECHqdZJaf=var_3)
        var_7, var_8, var_9 = storage.call_query(
            op=ops_by_name["RvEJgwBuNO"], inputs={"JECHqdZJaf": var_3}
        )
        # df = q.get_table(var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9)
        df = q.get_table(var_0, var_3, var_4, var_5, var_6, var_7, var_8, var_9)


def segfault():
    storage = Storage()

    @op
    def f(x: int) -> Tuple[int, int, int]:
        return x + 1, x + 1, x + 1

    with storage.query() as q:
        var_0 = Q()
        var_3 = Q()
        var_4, var_5, var_6 = f(var_0)
        var_7, var_8, var_9 = f(var_3)
        df = q.get_table(var_0, var_3, var_4, var_5, var_6, var_7, var_8, var_9)
