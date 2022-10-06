a = """
SELECT
"_140345709422496"."__uid__","_140345709421008"."__uid__","_140345709467184"."__uid__","_140345709425040"."__uid__","_140345709421728"."__uid__","_140345709421968"."__uid__","_140345709428016"."__uid__","_140345709414960"."__uid__"
FROM "RvEJgwBuNO_0" "_140345709428256","pUuTemKopZ_0"
"_140345709424656","__vrefs__" "_140345709422496","__vrefs__"
"_140345709421008","__vrefs__" "_140345709467184","__vrefs__"
"_140345709425040","__vrefs__" "_140345709421728","__vrefs__"
"_140345709421968","__vrefs__" "_140345709428016","__vrefs__" "_140345709414960"
WHERE "_140345709422496"."__uid__"="_140345709428256"."JECHqdZJaf" AND
"_140345709425040"."__uid__"="_140345709428256"."output_0" AND
"_140345709421728"."__uid__"="_140345709428256"."output_1" AND
"_140345709421968"."__uid__"="_140345709428256"."output_2" AND
"_140345709421008"."__uid__"="_140345709424656"."ZICffuGFgt" AND
"_140345709428016"."__uid__"="_140345709424656"."output_0" AND
"_140345709414960"."__uid__"="_140345709424656"."output_1"
"""

a = a.replace("_140345709422496", "a")
a = a.replace("_140345709421008", "b")
a = a.replace("_140345709467184", "c")
a = a.replace("_140345709425040", "d")
a = a.replace("_140345709421728", "e")
a = a.replace("_140345709421968", "f")
a = a.replace("_140345709428016", "g")
a = a.replace("_140345709414960", "h")
a = a.replace("_140345709428256", "i")
a = a.replace("_140345709424656", "j")
a = a.replace("RvEJgwBuNO_0", "k")
a = a.replace("pUuTemKopZ_0", "l")
a = a.replace("JECHqdZJaf", "m")
a = a.replace("ZICffuGFgt", "n")


b = """
SELECT
"a"."__uid__","b"."__uid__","c"."__uid__","d"."__uid__","e"."__uid__","f"."__uid__","g"."__uid__","h"."__uid__"
FROM "k" "i","l"
"j","__vrefs__" "a","__vrefs__"
"b","__vrefs__" "c","__vrefs__"
"d","__vrefs__" "e","__vrefs__"
"f","__vrefs__" "g","__vrefs__" "h"
WHERE "a"."__uid__"="i"."m" AND
"d"."__uid__"="i"."output_0" AND
"e"."__uid__"="i"."output_1" AND
"f"."__uid__"="i"."output_2" AND
"b"."__uid__"="j"."n" AND
"g"."__uid__"="j"."output_0" AND
"h"."__uid__"="j"."output_1"
"""

################################################################################
workflow = """
var_0 = Query()
var_1 = Query()
 = TYQOaNFvpU(CffuGFgtJs=var_0, TemKopZjZI=var_0)
 = yWAcqGFzYt(wLnGisiWgN=var_0, ZqITZMjtgU=var_0)
 = TYQOaNFvpU(CffuGFgtJs=var_0, TemKopZjZI=var_0)
var_2 = HZpnRLALrC(LOvmpbUrhT=var_0, QPSYwfuNhF=var_0)
 = TYQOaNFvpU(CffuGFgtJs=var_0, TemKopZjZI=var_0)
var_3, var_4, var_5 = RvEJgwBuNO(JECHqdZJaf=var_1)
 = yWAcqGFzYt(wLnGisiWgN=var_4, ZqITZMjtgU=var_1)
var_6 = fMZyuKpslm(cNQqEefRWi=var_2)
"""

workflow = workflow.replace("TYQOaNFvpU", "a")
workflow = workflow.replace("yWAcqGFzYt", "b")
workflow = workflow.replace("HZpnRLALrC", "c")
workflow = workflow.replace("RvEJgwBuNO", "d")
workflow = workflow.replace("fMZyuKpslm", "e")
workflow = workflow.replace("CffuGFgtJs", "f")
workflow = workflow.replace("TemKopZjZI", "g")
workflow = workflow.replace("LOvmpbUrhT", "h")
workflow = workflow.replace("QPSYwfuNhF", "i")
workflow = workflow.replace("JECHqdZJaf", "j")
workflow = workflow.replace("wLnGisiWgN", "k")
workflow = workflow.replace("ZqITZMjtgU", "l")
workflow = workflow.replace("cNQqEefRWi", "m")
