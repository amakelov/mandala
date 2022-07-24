from mandala_lite.all import *
from mandala_lite.tests.utils import *


def test_basics():
    storage = Storage()

    @op(storage=storage)
    def inc(x: int) -> int:
        return x + 1

    @op(storage=storage)
    def add(x: int, y: int) -> int:
        return x + y

    with run(storage):
        for i in range(20, 21):
            j = inc(i)
            final = add(i, j)

    with query(storage) as q:
        i = Query().named("i")
        j = inc(i).named("j")
        final = add(i, j).named('final')
        df = q.get_table(i, j, final)
        assert set(df["i"]) == {i for i in range(20, 25)}
        assert all(df["j"] == df["i"] + 1)
    check_invariants(storage)

SELECT "i"."__uid__","j"."__uid__","final"."__uid__" FROM "inc" "inc","add" "add","__vrefs__" "i","__vrefs__" "j","__vrefs__" "final" WHERE "i"."__uid__"="inc"."x" AND "j"."__uid__"="inc"."output_0" AND "i"."__uid__"="add"."x" AND "j"."__uid__"="add"."y" AND "final"."__uid__"="add"."output_0"

SELECT "i"."__uid__","j"."__uid__","final"."__uid__" FROM "inc" "inc","add" "add","__vrefs__" "i","__vrefs__" "j","__vrefs__" "final" WHERE "i"."__uid__"="inc"."x" AND "j"."__uid__"="inc"."output_0" AND "i"."__uid__"="add"."x" AND "j"."__uid__"="add"."y" AND "final"."__uid__"="add"."output_0"