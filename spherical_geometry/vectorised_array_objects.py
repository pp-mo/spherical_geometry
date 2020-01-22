'''
Created on 5 May 2015

@author: ppeglar
'''
# Control validity checking (turn off for speed)
ENABLE_CHECKING = True

class VectorisedArrayObject(object):
    """
    An abstract class for objects containing a group of coupled array-like
    properties.

    Provides a structure for array-specific operations, so that operations can
    be vectorized within a 'vectorised' object, instead of looped over an array
    of objects.
 
    Provides a group of array attributes, of known names, which share a common
    shape prefix which is the 'shape' of the object.
    Supports operations (distributed over components):
       __getitem__, reshape, copy
    operating over all component arrays to define a new object.
    Derived objects' components may be views on the original (as normal numpy).
    Additional generic array operations can be defined via helper methods.

    """
    # Define the array properties for this type (set in subclasses).
    # These will all exist as object properties, but some can be None.
    array_names = ()

    def __init__(self, arrays={}, shape=None):
        new_arrays = {key: None for key in self.array_names}
        new_arrays.update(arrays)
        self.set_arrays(new_arrays, shape=shape)

    def set_arrays(self, arrays_dict, shape=None):
        """
        Set array properties from a name-keyed dictionary.

        If shape is given, this overrides the automatic choice, which is the
        longest common to all arrays.

        """
        if ENABLE_CHECKING:
            # Check that all the arrays are in our recognised list
            assert all(key in self.array_names for key in arrays_dict)
        # Assign arrays from given list
        self.__dict__.update(arrays_dict)
        if shape == None:
            # Recalculate the longest-common-prefix shape
            arrays = [self.__dict__.get(key, None) for key in self.array_names]
            shapes_bylen = sorted([a.shape for a in arrays if a is not None],
                                  key=len)
            shape = shapes_bylen[0]
        self.shape = shape
        self.ndim = len(shape)
        if ENABLE_CHECKING:
            self._check_shapes_consistent()

    def _check_shapes_consistent(self):
        """Check that arrays match the common base shape."""
        arrays = [self.__dict__.get(key, None) for key in self.array_names]
        assert all(a.shape[:self.ndim] == self.shape
                   for a in arrays if a is not None)

    @classmethod
    def new_from_arrays(cls, arrays_dict, shape=None):
        """Make a new instance from a name-keyed dictionary of arrays."""
        return cls(arrays=arrays_dict, shape=shape)

    def new_by_function(self, function, new_shape=None,
                        *op_args, **op_kwargs):
        """Make new result by applying a common unary function to all the arrays."""
        arrays = {key: self.__dict__.get(key, None)
                  for key in self.array_names}
        new_arrays = {key: function(a, *op_args, **op_kwargs)
                      if a is not None else None
                      for key, a in arrays.items()}
        return self.new_from_arrays(new_arrays, shape=new_shape)

    def new_by_method(self, method_name, *args, **kwargs):
        """Make new result by calling a named unary method on all the arrays."""
        shape = kwargs.pop('new_by_method_shape', None)
        arrays = {key: self.__dict__.get(key, None)
                  for key in self.array_names}
        new_arrays = {key: (getattr(a, method_name)(*args, **kwargs)
                            if a is not None else None)
                      for key, a in arrays.items()}
        return self.new_from_arrays(new_arrays, shape=shape)

    def copy(self):
        return self.new_by_method('copy', new_by_method_shape=self.shape)

    def __getitem__(self, indices):
        return self.new_by_method('__getitem__', indices)

    def reshape(self, shape):
        arrays = {}
        for name in self.array_names:
            array = self.__dict__.get(name)
            if array is not None:
                a_shape = list(array.shape)
                a_shape = list(shape) + list(array.shape)[self.ndim:]
                arrays[name] = array.reshape(a_shape)
        return self.new_from_arrays(arrays, shape)

class TestVecObj(VectorisedArrayObject):
    array_names = ('a', 'b', 'c')


def tests():
    import numpy as np
    
    #testing...
    a1 = TestVecObj({'a':np.arange(24).reshape((3, 4, 2)) + 1001,
                     'b':np.arange(12).reshape((3, 4)) + 2001})
    
    print('a1.shape = {}'.format(a1.shape))
    print('a1.a.shape = {}'.format(a1.a.shape))
    print('a1.a = ')
    print(a1.a)
    print('a1.b.shape = {}'.format(a1.b.shape))
    print('a1.b =')
    print(a1.b)
    
    a2 = a1.reshape((12,))
    a2 = a1.copy()
    a2 = a1[1:4]
    a2 = a1[:, None]
    
    print('a2.shape = {}'.format(a2.shape))
    print('a2.a.shape = {}'.format(a2.a.shape))
    print('a2.a = ')
    print(a2.a)
    print('a2.b.shape = {}'.format(a2.b.shape))
    print('a2.b =')
    print(a2.b)
    
    a2.b[1:3] = 77
    print('new a2.b = ')
    print(a2.b)
    print('new a1.b = ')
    print(a1.b)

if __name__ == '__main__':
    tests()
