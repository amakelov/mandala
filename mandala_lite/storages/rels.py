from collections import defaultdict

from ..common_imports import *
from ..core.config import Config, Prov
from ..core.model import Call

def upsert_df(current:pd.DataFrame, new:pd.DataFrame) -> pd.DataFrame:
    return pd.concat([current, new[~new.index.isin(current.index)]])

class RelStorage:
    """
    Responsible for the low-level (i.e., unaware of mandala-specific concepts)
    interactions with the RDBMS part of the storage, such as creating and
    extending tables, running queries, etc. This is intended to be a pretty
    generic database interface supporting just the things we need.
    """
    def __init__(self):
        # internal name -> dataframe
        self.relations:Dict[str, pd.DataFrame] = {}

    ### 
    def create_relation(self, name:str, columns:List[str]):
        assert Config.uid_col in columns
        self.relations[name] = pd.DataFrame(columns=columns).set_index(Config.uid_col)
    
    def delete_relation(self, name:str):
        del self.relations[name]

    def create_column(self, relation:str, name:str, default_value:str):
        assert name not in self.relations[relation].columns
        self.relations[relation][name] = default_value
        
    ### 
    def insert(self, name:str, df:pd.DataFrame):
        self.relations[name] = self.relations[name].append(df.set_index(Config.uid_col),
                                                           verify_integrity=True)
    
    def upsert(self, name:str, df:pd.DataFrame):
        current = self.relations[name]
        df = df.set_index(Config.uid_col)
        self.relations[name] = upsert_df(current=current, new=df)

    def delete(self, name:str, index:List[str]):
        self.relations[name] = self.relations[name].drop(labels=index)

    ### 
    def select(self, query:Any) -> pd.DataFrame:
        raise NotImplementedError()
    

class RelAdapter:
    """
    Responsible for high-level RDBMS interactions, such as taking a bunch of
    calls and putting their data inside the database; uses `RelStorage` to do
    the actual work. 
    """
    def __init__(self, rel_storage:RelStorage):
        self.rel_storage = rel_storage 
        # initialize provenance table
        self.prov_df = pd.DataFrame(columns=[Prov.call_uid, Prov.op_name, Prov.op_version,
                                             Prov.is_super, Prov.vref_name, Prov.vref_uid,
                                             Prov.is_input])
        self.prov_df.set_index([Prov.call_uid, Prov.vref_name, Prov.is_input])

    @staticmethod
    def tabulate_calls(calls:List[Call]) -> Dict[str, pd.DataFrame]:
        # split by operation internal name
        calls_by_op = defaultdict(list)
        for call in calls:
            calls_by_op[call.op.sig.internal_name].append(call)
        res = {}
        for k, v in calls_by_op.items():
            res[k] = pd.DataFrame([{Config.uid_col: call.uid, 
                            **{k: v.uid for k, v in call.inputs.items()}, 
                            **{f'output_{i}': v.uid for i, v in enumerate(call.outputs)}}
                                   for call in v])
        return res
    
    def upsert_calls(self, calls:List[Call]):
        if calls:
            for name, df in self.tabulate_calls(calls).items():
                self.rel_storage.upsert(name, df)
            # update provenance table
            new_prov_df = self.get_provenance_table(calls=calls)
            self.prov_df = upsert_df(current=self.prov_df, new=new_prov_df)

    @staticmethod
    def get_provenance_table(calls:List[Call]) -> pd.DataFrame:
        dfs = []
        for call in calls:
            call_uid = call.uid
            op_name = call.op.sig.internal_name
            op_version = call.op.sig.version
            input_names = list(call.inputs.keys())
            input_uids = [call.inputs[k].uid for k in input_names]
            in_table = pd.DataFrame({Prov.call_uid: call_uid,
                                    Prov.op_name: op_name,
                                    Prov.op_version: op_version,
                                    Prov.is_super: call.op.sig.is_super,
                                    Prov.vref_name: input_names,
                                    Prov.vref_uid: input_uids,
                                    Prov.is_input: True, 
                                    })
            output_names = list([f'output_{i}' for i in range(len(call.outputs))])
            output_uids = [call.outputs[i].uid for i in range(len(call.outputs))]
            out_table = pd.DataFrame({Prov.call_uid: call_uid,
                                    Prov.op_name: op_name,
                                    Prov.op_version: op_version,
                                    Prov.is_super: call.op.sig.is_super,
                                    Prov.vref_name: output_names,
                                    Prov.vref_uid: output_uids,
                                    Prov.is_input: False, 
                                    })
            df = pd.concat([in_table, out_table], ignore_index=True)       
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)