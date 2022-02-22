from ..common_imports import *

def query_yes_no(question:str, data:TAny=None, default:str='no') -> bool:
    options = {
        'yes': True,
        'y': True,
        'no': False,
        'n': False
    }
    if default is None:
        prompt = '[y/n]'
    elif default == 'yes':
        prompt = '[Y/n]'
    elif default == 'no':
        prompt = '[y/N]'
    else:
        raise ValueError(f'Got invalid default answer {default}')
    full_prompt = f'{question} {prompt}\n{repr(data)}'
    while True:
        sys.stdout.write(full_prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return options[default]
        elif choice in options:
            return options[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

# as a decorator
class CascadingVerifier(object):
    
    def __init__(self, question:str, desc_getter:TCallable=None):
        self.question = question
        self.desc_getter = desc_getter
    
    def __call__(self, method:TCallable) -> 'method':
        @functools.wraps(method)
        def f(instance:TAny, *args, answer:bool=None, **kwargs):
            if answer is None:
                if self.desc_getter is not None:
                    aux_data = self.desc_getter(instance)
                else:
                    aux_data = instance
                answer = query_yes_no(question=self.question, data=aux_data)
                if not answer:
                    return None
                else:
                    result = method(instance, *args, answer=answer, **kwargs)
                    return result
            else:
                assert answer == True
                result = method(instance, *args, answer=answer, **kwargs)
                return result
        return f

ask = CascadingVerifier