from ..common_imports import *

################################################################################
### optional dependencies
################################################################################
try:
    import psycopg2
except ImportError:
    _has_psql = False
else:
    _has_psql = True

try:
    import dask
except ImportError:
    _has_dask = False
else:
    _has_dask = True

try:
    import ray
except ImportError:
    _has_ray = False
else:
    _has_ray = True

try:
    import sklearn
except ImportError:
    _has_sklearn = False
else:
    _has_sklearn = True

try:
    import rich
except ImportError:
    _has_rich = False
else:
    _has_rich = True


class EnvConfig(object):
    has_psql = _has_psql
    has_dask = _has_dask
    has_ray = _has_ray
    has_sklearn = _has_sklearn
    has_rich = _has_rich

################################################################################
### logging helper
################################################################################
class LoggingLevels(object):
    debug = 10
    info = 20
    warning = 30
    error = 40
    critical = 50

################################################################################
### important lib behaviors
################################################################################
class PSQLConfig(object):
    
    def __init__(self, **kwargs):
        self.set(**kwargs)
    
    def set(self, user:str='postgres', host:str='localhost', 
            port:int=5432, root_db_name:str='postgres', password:str='postgres'):
        """
        Settings for connecting to a postgresql server
        """
        self.user = user
        self.host = host
        self.port = port
        self.root_db_name = root_db_name
        self.password = password
    
    def __repr__(self):
        return f'PSQLConfig({self.__dict__})'
    

class SuperopWrapping(object):
    legacy = 'legacy' # old
    construct_and_deconstruct = 'construct_and_deconstruct' # newer
    construct_only = 'construct_only' # best


class Config(object):
    def __init__(self, 
                 db_backend:str='sqlite',
                 fs_storage_root:Path=None,
                 test_output_root:Path=None,
                 default_hash_method:str='causal',
                 enable_vref_magics:bool=True,
                 enable_query_magics:bool=True,
                 enable_autonesting:bool=True,
                 strict_signatures:bool=False,
                 enable_defaults:bool=True,
                 bind_defaults_in_queries:bool=False,
                 allow_new_inputs:bool=True,
                 autowrap:bool=True,
                 require_returns_unwrapped:bool=False,
                 on_var_signature:str='skip',
                 track_provenance:bool=True,
                 build_call_graph:bool=True,
                 include_constructive_indexing_in_rels:bool=False,
                 superop_wrapping_style:str=SuperopWrapping.construct_only,
                 decompose_struct_as_many:bool=False,
                 autocommit:bool=True,
                 autodelete:bool=True,
                 logging_level:str='info',
                 transaction_loglevel:int=LoggingLevels.debug,
                 verbose_deletes:bool=True,
                 verbose_commits:bool=True,
                 use_rich:bool=True):
        """
        Args:
            db_backend (str, optional): one of ('sqlite', 'psql')
            fs_storage_root (Path, optional): root directory for storages
            test_output_root (Path, optional): root directory for test output files
            default_hash_method (str, optional): one of ('causal', 'content')
            enable_vref_magics (bool, optional): allow unary/binary ops to apply
            to atom vrefs
            enable_query_magics (bool, optional): allow comparison operators to induce
            constraints on queries to variables
            enable_autonesting (bool, optional): allow top-level contexts to merge
            into an existing global context
            strict_signatures (bool, optional): require type annotations to be Type
            instances (as opposed to regular Python annotations)
            enable_defaults (bool, optional): allow op defs to contain
            defaults and bind defaults in op calls in `run` mode
            bind_defaults_in_queries (bool, optional): impose constraints in
            queries when no value is passed
            allow_new_inputs (bool, optional): allow creation of new inputs
            within same version of the op
            autowrap (bool, optional): whether ops can automatically wrap their
            inputs 
            require_returns_unwrapped (bool, optional): if values passed to
            __returns__ must be fully unwrapped
            on_var_signature (str, optional): how to handle non-fixed signatures
            track_provenance (bool, optional): put calls in provenance table
            build_call_graph (bool, optional): keep track of which superops call
            which other ops directly
            include_constructive_indexing_in_rels (bool, optional): unused
            superop_wrapping_style (str, optional): one of SuperopWrapping attrs
            decompose_struct_as_many (bool, optional): use multiple
            deconstructive calls for structs (as opposed to 1)
            autocommit (bool, optional): commit calls to relations upon exiting
            `run` context
            autodelete (bool, optional): delete data upon exiting `delete` context
            logging_level (str, optional): Defaults to 'info'.
            transaction_loglevel (int, optional): 
            verbose_deletes (bool, optional): print out summary of what got deleted
            verbose_commits (bool, optional): print out summary of what got committed
            use_rich (bool, optional): use rich for pretty printing
        """
        self._db_backend = db_backend
        self._fs_storage_root = fs_storage_root
        self._test_output_root = test_output_root 
        self._default_hash_method = default_hash_method 
        self._enable_vref_magics = enable_vref_magics 
        self._enable_query_magics = enable_query_magics 
        self._enable_autonesting = enable_autonesting 
        self._strict_signatures = strict_signatures 
        self._enable_defaults = enable_defaults 
        self._bind_defaults_in_queries = bind_defaults_in_queries 
        self._allow_new_inputs = allow_new_inputs 
        self._autowrap = autowrap 
        self._require_returns_unwrapped = require_returns_unwrapped 
        self._on_var_signature = on_var_signature 
        self._track_provenance = track_provenance 
        self._build_call_graph = build_call_graph 
        self._include_constructive_indexing_in_rels = include_constructive_indexing_in_rels 
        self._superop_wrapping_style = superop_wrapping_style 
        self._decompose_struct_as_many = decompose_struct_as_many 
        self._autocommit = autocommit 
        self._autodelete = autodelete 
        self._logging_level = logging_level 
        self._transaction_loglevel = transaction_loglevel 
        self._verbose_deletes = verbose_deletes 
        self._verbose_commits = verbose_commits 
        self._use_rich = use_rich 

        if EnvConfig.has_psql:
            self._psql = PSQLConfig()
    
    @property
    def db_backend(self) -> str:
        return self._db_backend
    
    @property
    def psql(self) -> PSQLConfig:
        return self._psql

    @property
    def fs_storage_root(self) -> TOption[Path]:
        return self._fs_storage_root
    
    def set_fs_storage_root(self, p:Path):
        self._fs_storage_root = p

    @property
    def test_output_root(self) -> TOption[Path]:
        return self._test_output_root

    def set_test_output_root(self, p:Path):
        self._test_output_root = p

    @property
    def default_hash_method(self) -> str:
        return self._default_hash_method
    
    def set_default_hash_method(self, value:str):
        self._default_hash_method = value

    @property
    def enable_vref_magics(self) -> bool:
        return self._enable_vref_magics
    
    def set_vref_magics(self, value:bool):
        self._enable_vref_magics = value

    @property
    def enable_query_magics(self) -> bool:
        return self._enable_query_magics
    
    def set_query_magics(self, value:bool):
        self._enable_query_magics = value

    @property
    def enable_autonesting(self) -> bool:
        return self._enable_autonesting
    
    def set_autonesting(self, value:bool):
        self._enable_autonesting = value
    
    @property
    def strict_signatures(self) -> bool:
        return self._strict_signatures
    
    def set_strict_signatures(self, value:bool):
        self._strict_signatures = value

    @property
    def enable_defaults(self) -> bool:
        return self._enable_defaults
    
    def set_defaults(self, value:bool):
        self._enable_defaults = value
    
    @property
    def bind_defaults_in_queries(self) -> bool:
        return self._bind_defaults_in_queries
    
    def set_bind_defaults_in_queries(self, value:bool):
        self._bind_defaults_in_queries = value
    
    @property
    def allow_new_inputs(self) -> bool:
        return self._allow_new_inputs
    
    def set_allow_new_inputs(self, value:bool):
        self._allow_new_inputs = value
    
    @property
    def autowrap(self) -> bool:
        return self._autowrap

    def set_autowrap(self, value:bool):
        self._autowrap = value
    
    @property
    def require_returns_unwrapped(self) -> bool:
        return self._require_returns_unwrapped
    
    def set_require_returns_unwrapped(self, value:bool):
        self._require_returns_unwrapped = value
    
    @property
    def on_var_signature(self) -> str:
        return self._on_var_signature
    
    def set_on_var_signature(self, value:str):
        self._on_var_signature = value

    @property
    def track_provenance(self) -> bool:
        return self._track_provenance
    
    def set_track_provenance(self, value:bool):
        self._track_provenance = value
    
    @property
    def build_call_graph(self) -> bool:
        return self._build_call_graph

    def set_build_call_graph(self, value:bool):
        self._build_call_graph = value
       
    @property
    def include_constructive_indexing_in_rels(self) -> bool:
        return self._include_constructive_indexing_in_rels
    
    def set_include_constructive_indexing(self, value:bool):
        self._include_constructive_indexing = value
    
    @property
    def superop_wrapping_style(self) -> str:
        return self._superop_wrapping_style

    def set_superop_wrapping_style(self, value:str):
        self._superop_wrapping_style = value
       
    @property
    def decompose_struct_as_many(self) -> bool:
        return self._decompose_struct_as_many

    def set_decompose_struct_as_many(self, value:bool):
        self._decompose_struct_as_many = value
    
    @property
    def autocommit(self) -> bool:
        return self._autocommit

    def set_autocommit(self, value:bool):
        self._autocommit = value
    
    @property
    def autodelete(self) -> bool:
        return self._autodelete

    def set_autodelete(self, value:bool):
        self._autodelete = value
    
    @property
    def logging_level(self) -> str:
        return self._logging_level

    def set_logging_level(self, value:str):
        self._logging_level = value
    
    @property
    def transaction_loglevel(self) -> int:
        return self._transaction_loglevel

    def set_transaction_loglevel(self, value:int):
        self._transaction_loglevel = value
    
    @property
    def verbose_deletes(self) -> bool:
        return self._verbose_deletes

    def set_verbose_deletes(self, value:bool):
        self._verbose_deletes = value
    
    @property
    def verbose_commits(self) -> bool:
        return self._verbose_commits

    def set_verbose_commits(self, value:bool):
        self._verbose_commits = value

    @property
    def use_rich(self) -> bool:
        return self._use_rich

    
CoreConfig = Config()
if CoreConfig.use_rich and EnvConfig.has_rich:
    from rich import pretty
    pretty.install()

################################################################################
### global constants
################################################################################
class CoreConsts(object):
    UID_COL = '__index__'
    PARTITION_COL = '__partition__'
    CONTEXT_KWARG = '__context__'
    APPLY_DEFAULTS_SKIP = (CONTEXT_KWARG, )

    class List(object):
        LIST = 'list'
        IDX = 'idx'
        ELT = 'elt'
    
    class Dict(object):
        DICT = 'dict'
        KEY = 'key'
        VALUE = 'value'


class CALLS(object):
    main_partition = '__main__'
    default_temp_partition = '__temp__'


class MODES(object):
    noop = 'noop'
    transient = 'transient'
    query = 'query'
    run = 'run'
    delete = 'delete'
    query_delete = 'query_delete'
    define = 'define'
    capture = 'capture'

    _modes = (
        noop, transient, query, run, delete, query_delete, define, capture
        )

    DEFAULT = noop