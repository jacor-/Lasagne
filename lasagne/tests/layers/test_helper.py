import pytest
import numpy as np
import theano


import lasagne


class TestGetAllLayers(object):
    def test_stack(self):
        from lasagne.layers import InputLayer, DenseLayer, get_all_layers
        from itertools import permutations
        # l1 --> l2 --> l3
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        for count in (0, 1, 2, 3):
            for query in permutations([l1, l2, l3], count):
                if l3 in query:
                    expected = [l1, l2, l3]
                elif l2 in query:
                    expected = [l1, l2]
                elif l1 in query:
                    expected = [l1]
                else:
                    expected = []
                assert get_all_layers(query) == expected

    def test_merge(self):
        from lasagne.layers import (InputLayer, DenseLayer, ElemwiseSumLayer,
                                    get_all_layers)
        # l1 --> l2 --> l3 --> l6
        #        l4 --> l5 ----^
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        l4 = InputLayer((10, 30))
        l5 = DenseLayer(l4, 40)
        l6 = ElemwiseSumLayer([l3, l5])
        assert get_all_layers(l6) == [l1, l2, l3, l4, l5, l6]
        assert get_all_layers([l4, l6]) == [l4, l1, l2, l3, l5, l6]
        assert get_all_layers([l5, l6]) == [l4, l5, l1, l2, l3, l6]
        assert get_all_layers([l4, l2, l5, l6]) == [l4, l1, l2, l5, l3, l6]

    def test_split(self):
        from lasagne.layers import InputLayer, DenseLayer, get_all_layers
        # l1 --> l2 --> l3
        #  \---> l4
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)
        l4 = DenseLayer(l1, 50)
        assert get_all_layers(l3) == [l1, l2, l3]
        assert get_all_layers(l4) == [l1, l4]
        assert get_all_layers([l3, l4]) == [l1, l2, l3, l4]
        assert get_all_layers([l4, l3]) == [l1, l4, l2, l3]

    def test_bridge(self):
        from lasagne.layers import (InputLayer, DenseLayer, ElemwiseSumLayer,
                                    get_all_layers)
        # l1 --> l2 --> l3 --> l4 --> l5
        #         \------------^
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 30)
        l4 = ElemwiseSumLayer([l2, l3])
        l5 = DenseLayer(l4, 40)
        assert get_all_layers(l5) == [l1, l2, l3, l4, l5]


class TestGetAllParams(object):
    def test_get_all_params(self):
        from lasagne.layers import (InputLayer, DenseLayer, get_all_params)
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)

        assert get_all_params(l3) == l2.get_params() + l3.get_params()
        assert (get_all_params(l3, regularizable=False) ==
                (l2.get_params(regularizable=False) +
                 l3.get_params(regularizable=False)))

        assert (get_all_params(l3, regularizable=True) ==
                (l2.get_params(regularizable=True) +
                 l3.get_params(regularizable=True)))


class TestCountParams(object):
    def test_get_all_params(self):
        from lasagne.layers import (InputLayer, DenseLayer, count_params)
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)

        num_weights = 20 * 30 + 30 * 40
        num_biases = 30 + 40

        assert count_params(l3, regularizable=True) == num_weights
        assert count_params(l3, regularizable=False) == num_biases
        assert count_params(l3) == num_weights + num_biases


class TestGetAllParamValues(object):
    def test_get_all_param_values(self):
        from lasagne.layers import (InputLayer, DenseLayer,
                                    get_all_param_values)
        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)

        pvs = get_all_param_values(l3)
        assert len(pvs) == 4


class TestSetAllParamValues(object):
    def test_set_all_param_values(self):
        from lasagne.layers import (InputLayer, DenseLayer,
                                    set_all_param_values)
        from lasagne.utils import floatX

        l1 = InputLayer((10, 20))
        l2 = DenseLayer(l1, 30)
        l3 = DenseLayer(l2, 40)

        a = floatX(np.random.normal(0, 1, (1, 1)))
        b = floatX(np.random.normal(0, 1, (1,)))
        set_all_param_values(l3, [a, b, a, b])
        assert np.allclose(l3.W.get_value(), a)
        assert np.allclose(l3.b.get_value(), b)
        assert np.allclose(l2.W.get_value(), a)
        assert np.allclose(l2.b.get_value(), b)

        with pytest.raises(ValueError):
            set_all_param_values(l3, [a, b, a])
