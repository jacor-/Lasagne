from mock import Mock
import numpy
import pytest
import theano


class TestLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock())

    @pytest.fixture
    def named_layer(self):
        from lasagne.layers.base import Layer
        return Layer(Mock(), name='layer_name')

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.get_output_shape()

    def test_get_output_shape(self, layer):
        assert layer.get_output_shape() == layer.input_layer.get_output_shape()

    def test_get_output_without_arguments(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output()
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output.return_value)
        layer.input_layer.get_output.assert_called_with(None)

    def test_get_output_passes_on_arguments_to_input_layer(self, layer):
        input, kwarg = object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with(
            layer.input_layer.get_output.return_value, kwarg=kwarg)
        layer.input_layer.get_output.assert_called_with(
            input, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1, 2, 3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)
        assert layer.get_output_shape() == (None, 20)

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        with pytest.raises(RuntimeError):
            layer.get_output()
        with pytest.raises(RuntimeError):
            layer.get_output(Mock())
        with pytest.raises(RuntimeError):
            layer.get_output({Mock(): Mock()})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_named_layer(self):
        from lasagne.layers.base import Layer
        l = Layer(Mock(), name="foo")
        assert l.name == "foo"

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_params_tags(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape, name='A', tag1=True, tag2=False)

        b_shape = (30, 20)
        b = numpy.random.normal(0, 1, b_shape)
        B = layer.add_param(b, b_shape, name='B', tag1=True, tag2=True)

        c_shape = (40, 10)
        c = numpy.random.normal(0, 1, c_shape)
        C = layer.add_param(c, c_shape, name='C', tag2=True)

        assert layer.get_params() == [A, B, C]
        assert layer.get_params(tag1=True) == [A, B]
        assert layer.get_params(tag1=False) == [C]
        assert layer.get_params(tag2=True) == [B, C]
        assert layer.get_params(tag2=False) == [A]
        assert layer.get_params(tag1=True, tag2=True) == [B]

    def test_add_param_tags(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape)
        assert A in layer.params
        assert 'trainable' in layer.params[A]
        assert 'regularizable' in layer.params[A]

        b_shape = (30, 20)
        b = numpy.random.normal(0, 1, b_shape)
        B = layer.add_param(b, b_shape, trainable=False)
        assert B in layer.params
        assert 'trainable' not in layer.params[B]
        assert 'regularizable' in layer.params[B]

        c_shape = (40, 10)
        c = numpy.random.normal(0, 1, c_shape)
        C = layer.add_param(c, c_shape, tag1=True)
        assert C in layer.params
        assert 'trainable' in layer.params[C]
        assert 'regularizable' in layer.params[C]
        assert 'tag1' in layer.params[C]

    def test_add_param_name(self, layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = layer.add_param(a, a_shape, name='A')
        assert A.name == 'A'

    def test_add_param_named_layer_name(self, named_layer):
        a_shape = (20, 50)
        a = numpy.random.normal(0, 1, a_shape)
        A = named_layer.add_param(a, a_shape, name='A')
        assert A.name == 'layer_name.A'


class TestMultipleInputsLayer:
    @pytest.fixture
    def layer(self):
        from lasagne.layers.base import MultipleInputsLayer
        return MultipleInputsLayer([Mock(), Mock()])

    def test_get_output_shape(self, layer):
        layer.get_output_shape_for = Mock()
        result = layer.get_output_shape()
        assert result is layer.get_output_shape_for.return_value
        layer.get_output_shape_for.assert_called_with([
            layer.input_layers[0].get_output_shape.return_value,
            layer.input_layers[1].get_output_shape.return_value,
            ])

    def test_get_output_without_arguments(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output()
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output.return_value,
            layer.input_layers[1].get_output.return_value,
            ])
        layer.input_layers[0].get_output.assert_called_with(None)
        layer.input_layers[1].get_output.assert_called_with(None)

    def test_get_output_passes_on_arguments_to_input_layer(self, layer):
        input, kwarg = object(), object()
        layer.get_output_for = Mock()

        output = layer.get_output(input, kwarg=kwarg)
        assert output is layer.get_output_for.return_value
        layer.get_output_for.assert_called_with([
            layer.input_layers[0].get_output.return_value,
            layer.input_layers[1].get_output.return_value,
            ], kwarg=kwarg)
        layer.input_layers[0].get_output.assert_called_with(
            input, kwarg=kwarg)
        layer.input_layers[1].get_output.assert_called_with(
            input, kwarg=kwarg)

    def test_get_output_input_is_a_mapping(self, layer):
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_output_input_is_a_mapping_no_key(self, layer):
        layer.get_output_for = Mock()

        output = layer.get_output({})
        assert output is layer.get_output_for.return_value

    def test_get_output_input_is_a_mapping_to_array(self, layer):
        input = {layer: [[1, 2, 3]]}
        output = layer.get_output(input)
        assert numpy.all(output.eval() == input[layer])

    @pytest.fixture
    def layer_from_shape(self):
        from lasagne.layers.base import MultipleInputsLayer
        return MultipleInputsLayer([(None, 20), Mock()])

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layers[0] is None
        assert layer.input_shapes[0] == (None, 20)
        shape1 = layer.input_layers[1].get_output_shape()
        assert layer.input_layers[1] is not None
        assert layer.input_shapes[1] == shape1
        layer.get_output_shape_for = Mock()
        result = layer.get_output_shape()
        assert result is layer.get_output_shape_for.return_value
        layer.get_output_shape_for.assert_called_with([
            layer.input_shapes[0],
            layer.input_layers[1].get_output_shape.return_value,
            ])

    def test_layer_from_shape_invalid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        with pytest.raises(RuntimeError):
            layer.get_output()
        with pytest.raises(RuntimeError):
            layer.get_output(Mock())
        with pytest.raises(RuntimeError):
            layer.get_output({layer.input_layers[1]: Mock()})

    def test_layer_from_shape_valid_get_output(self, layer_from_shape):
        layer = layer_from_shape
        input = {layer: theano.tensor.matrix()}
        assert layer.get_output(input) is input[layer]

    def test_get_params(self, layer):
        assert layer.get_params() == []
