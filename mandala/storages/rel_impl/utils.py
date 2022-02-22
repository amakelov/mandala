from sqlalchemy import MetaData
from sqlalchemy.engine.base import Connection
from sqlalchemy.sql.elements import TextClause

from ...common_imports import *
from ...core.config import CoreConfig


class transaction(object):
    """
    Used to decorate methods of classes to turn them into transactions. 
    
    Your class should support the following things:
        - a .get_engine() method, which returns a sqlalchemy engine
        - a .block_subtransactions attribute

    NOTE: when chaining @transaction methods, it is your responsibility to
    provide the conn argument when needed.
    """
    def __init__(self, retry:bool=False, retry_mean_interval:float=5.0,
                 max_retries:int=None):
        self.retry = retry
        self.retry_mean_interval = retry_mean_interval
        self.max_retries = max_retries
    
    def __call__(self, method) -> 'method':

        @functools.wraps(method)
        def inner(instance:TAny, *args, conn:Connection=None, **kwargs):
            if conn is None:
                # new transaction
                if instance.block_subtransactions:
                    raise RuntimeError()
                logging.log(
                    CoreConfig.transaction_loglevel,
                    f'Starting new transaction from method {method.__name__}'
                )
                if not self.retry:
                    with instance.get_engine().begin() as conn:
                        result = method(instance, *args, conn=conn, **kwargs)
                        logging.log(
                            CoreConfig.transaction_loglevel,
                            f'Transaction finished from function {method}.'
                        )
                    return result
                else:
                    success = False
                    retries = 0
                    max_retries = self.max_retries if self.max_retries is not None else math.inf
                    while (not success) and (retries < max_retries):
                        try:
                            with instance.get_engine().begin() as conn:
                                result = method(instance, *args, conn=conn, **kwargs)
                                logging.log(
                                    CoreConfig.transaction_loglevel,
                                    f'Transaction finished from function {method}.'
                                )
                            success = True
                            return result
                        except Exception as e:
                            print(e)
                            logging.debug('Retrying to get a connection...')
                            retry_interval = np.random.uniform(low=-1.0, high=1.0) + self.retry_mean_interval
                            time.sleep(retry_interval)
                            retries += 1
                    raise RuntimeError('Max retries exceeded')
            else:
                # nest in existing transaction
                logging.debug(f'Folding {method.__name__} into current transaction')
                result = method(instance, *args, conn=conn, **kwargs)
                return result

        return inner

def get_engine(connection_string:str, autocommit:bool=False, 
               timeout:int=None, creator:TCallable=None):
    if timeout is not None:
        connect_args = {'connect_timeout': 5}
    else:
        connect_args = {}
    args = (connection_string,)
    kwargs:TDict[str, TAny] = {
        'connect_args': connect_args
    }
    if autocommit:
        kwargs['isolation_level'] = 'AUTOCOMMIT'
    if creator is not None:
        kwargs['creator'] = creator
    engine = sqlalchemy.create_engine(*args, **kwargs)
    return engine

def get_metadata(engine=None, conn:Connection=None):
    if engine is not None:
        return MetaData(bind=engine)
    elif conn is not None:
        return MetaData(bind=conn)
    else:
        raise ValueError()

def get_in_clause_rhs(index_like:TUnion[pd.Index, TIter],
                      ) -> TUnion[list, TextClause]:
    if not isinstance(index_like, pd.Index):
        elts = list(index_like)
    else:
        elts = index_like.values.tolist()
    if not elts:
        return []
    else:
        return tuple(elts)
        # return sql.text(', '.join([f"'{elt}'" for elt in elts]))