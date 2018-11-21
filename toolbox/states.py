import signal
from multiprocessing import current_process

import torch


class State:
    def __init__(self, attrib, dump_fn=lambda x: x, load_fn=lambda x: x):
        """
        Wrap this around the attribute you want to track in __init__ of a
        Trackable subclass
        Args:
            attrib: The attribute you want to track
            dump_fn: The function for converting the attribute to a serializable
                     form
            load_fn: Convert the serializable form back to the original value
        """
        self._attrib = attrib
        self._dump_fn = dump_fn
        self._load_fn = load_fn

    @property
    def attrib(self):
        return self._attrib

    def dump(self, curr_val):
        return self._dump_fn(curr_val)

    def load(self, dumped_val):
        return self._load_fn(dumped_val)


class TorchState(State):
    """
    A more specialized state wrapper for PyTorch objects who have state_dict()
    function.
    """

    def __init__(self, attrib):
        def load_fn(state_dict):
            attrib.load_state_dict(state_dict)
            return attrib

        super().__init__(attrib, dump_fn=lambda x: x.state_dict(),
                         load_fn=load_fn)


class TrackableMeta(type):
    """
    Metaclass for registering objects.
    """

    def __new__(cls, cls_name, bases, attr):
        state_dict = {}

        def set_attr(self, attr_name, value):
            if issubclass(value.__class__, State):
                state_dict[attr_name] = value
                value = value.attrib
            object.__setattr__(self, attr_name, value)

        def save_state(self, save_path=None):
            attrs = self.__dict__
            out_dict = {}
            for attr_name, state_obj in state_dict.items():
                value = state_obj.dump(attrs[attr_name])
                out_dict[attr_name] = value
            if save_path is not None:
                torch.save(out_dict, save_path)
            return out_dict

        def load_state(self, in_dict, ignore_missing=False):
            for attr_name, value in in_dict.items():
                if not (attr_name in state_dict):
                    print('attribute %s is not part of the object' % attr_name)
                    if ignore_missing:
                        print('abort loading')
                        return
                value = state_dict[attr_name].load(value)
                self.__dict__[attr_name] = value
            self.reloaded = True

        attr['__setattr__'] = set_attr
        attr['save_state'] = save_state
        attr['load_state'] = load_state
        attr['reloaded'] = False
        return super().__new__(cls, cls_name, bases, attr)


class Trackable(State, metaclass=TrackableMeta):
    """
    The Trackable base class. If you want to automatically save the state of the
    attributes, you should subclass this class. For each tracked state, wrap it
    with a State object when initialized in __init__.
    """

    @property
    def attrib(self):
        return self

    def dump(self, curr_val):
        return curr_val.save_state()

    def load(self, dumped_val):
        self.load_state(dumped_val)
        return self


def save_on_interrupt(save_path=None, exception_handling=None):
    """
    The save on interrupt decorator. For any (member) function decorated, it
    will capture SIG_INT and save the state.
    Args:
        save_path: The path for saving the state
        exception_handling: A function with only parameter self. Called when
        signal is received.

    Returns: The decorated function contains the original function.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            process_name = current_process().name

            def handler(sig, frame):
                print(self.__class__)
                if current_process().name == process_name:
                    if exception_handling is not None:
                        exception_handling(self)
                    self.save_state(save_path)
                    raise KeyboardInterrupt()

            top_level = 'nested_tracked_state' not in globals()
            if top_level:
                global nested_tracked_state
                nested_tracked_state = True
                old_handler = signal.signal(signal.SIGINT, handler)
                out = None
                try:
                    out = func(self, *args, **kwargs)
                except KeyboardInterrupt:
                    pass
                signal.signal(signal.SIGINT, old_handler)
                del nested_tracked_state
                return out
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
