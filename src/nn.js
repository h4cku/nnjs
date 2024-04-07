// Activation Functions
class ActivationFunction {
    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    static tanh(x) {
        return (1 - Math.exp(-x)) / (1 + Math.exp(-x));
    }

    static relu(x) {
        return x > 0 ? x : 0;
    }

    static gelu(x) {
        return x * ActivationFunction.sigmoid(1.702 * x);
    }

    static silu(x) {
        return x * ActivationFunction.sigmoid(x);
    }
}

// Tensor class
class Tensor {
    constructor(data, shape) {
        this.data = data;
        this.shape = shape;
        this.stride = Array(this.shape.length);
        this.stride_ = Array(this.shape.length);
        let acc = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            this.stride_[i] = acc; // this will keep the real striding
            if (this.shape[i] == 1) this.stride[i] = 0; // to allow broadcasting
            else this.stride[i] = acc;
            acc *= this.shape[i];
        }
    }

    item() {
        return this.data[0];
    }

    match_dimension(m) {
        if (this.shape.length != m.shape.length) {
            return false;
        }
        for (let i = 0; i < this.shape.length; i++) {
            if (m.stride[i] != 0 && this.shape[i] != m.shape[i]) {
                return false;
            }
        }
        return true;
    }

    reshape(new_shape) {
        this.shape = new_shape;
        this.stride = Array(this.shape.length);
        let acc = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            this.stride[i] = acc;
            acc *= this.shape[i];
        }
    }

    set_stride(new_stride) {
        this.stride = new_stride;
    }

    random() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = Math.random();
        }
    }

    zeros() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = 0;
        }
    }

    ones() {
        for (let i = 0; i < this.data.length; i++) {
            this.data[i] = 1;
        }
    }

    at(idx) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        return this.data[s];
    }

    set(idx, val) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        this.data[s] = val;
    }

    get_idx(i) {
        let idx = [];
        for (let j = 0; j < this.stride.length; j++) {
            let t_ = Math.floor(i / this.stride_[j]);
            i = i - t_ * this.stride[j];
            idx.push(t_);
        }
        return idx;
    }

    get_pos(idx) {
        let s = 0;
        for (let i = 0; i < idx.length; i++) {
            s += idx[i] * this.stride[i];
        }
        return s;
    }

    get_dim() {
        let acc = 1;
        for (let i = 0; i < this.shape.length; i++) {
            acc *= this.shape[i];
        }
        return acc;
    }

    apply_unitary_op(f) {
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(f(this.data[i]));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    apply_binary_op(m, f) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(f(this.data[i], m.at(idx)));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    sum(axis) {
        let new_shape = [...this.shape];
        new_shape[axis] = 1;
        let o = new Tensor(Array(this.get_dim() / this.shape[axis]), new_shape);
        o.zeros();
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            idx[axis] = 0;
            o.data[o.get_pos(idx)] += this.data[i];
        }
        return o;
    }

    mean(axis) {
        let o = this.sum(axis);
        o = o.times(1 / this.shape[axis]);
        return o;
    }

    add(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] + m.at(idx));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    sub(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] - m.at(idx));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    mul(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] * m.at(idx));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    div(m) {
        if (!this.match_dimension(m)) {
            return null;
        }
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            let idx = this.get_idx(i);
            new_data.push(this.data[i] / m.at(idx));
        }
        return new Tensor(new_data, [...this.shape]);
    }

    times(s) {
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(this.data[i] * s);
        }
        return new Tensor(new_data, [...this.shape]);
    }

    dot(m) {
        // TO IMPROVE
        let o = new Tensor(Array(this.shape[0] * m.shape[1]), [
            this.shape[0],
            m.shape[1],
        ]);
        for (let i = 0; i < this.shape[0]; i++) {
            for (let j = 0; j < m.shape[1]; j++) {
                let s = 0;
                for (let k = 0; k < this.shape[1]; k++) {
                    s += this.at([i, k]) * m.at([k, j]);
                }
                o.set([i, j], s);
            }
        }
        return o;
    }

    transpose(i, j) {
        let o = new Tensor(this.data, [...this.shape]);

        let x_ = o.stride[i];
        o.stride[i] = o.stride[j];
        o.stride[j] = x_;

        x_ = o.shape[i];
        o.shape[i] = o.shape[j];
        o.shape[j] = x_;

        return o;
    }
    pow(e) {
        let new_data = [];
        for (let i = 0; i < this.data.length; i++) {
            new_data.push(Math.pow(this.data[i], e));
        }
        return new Tensor(new_data, [...this.shape]);
    }
}

// Backward Classes
class NopBackward {
    constructor() {}
    call(loss) {}
}

class AddBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss);
        // Shrink over broadcast axis
        for (let i = 0; i < this.y.val.stride.length; i++) {
            if (this.y.val.stride[i] == 0) {
                loss = loss.sum(i);
            }
        }
        this.y.backward(loss);
    }
}

class SubBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss);
        for (let i = 0; i < this.y.val.stride.length; i++) {
            if (this.y.val.stride[i] == 0) {
                loss = loss.sum(i);
            }
        }
        this.y.backward(loss.times(-1));
    }
}

class MulBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss.mul(y.val));
        loss = loss.mul(x.val);
        for (let i = 0; i < this.y.val.stride.length; i++) {
            if (this.y.val.stride[i] == 0) {
                loss = loss.sum(i);
            }
        }
        this.y.backward(loss);
    }
}

class DivBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    helper(a, b) {
        return -a / b ** 2;
    }
    call(loss) {
        this.x.backward(loss.div(this.y.val));
        loss = loss.mul(this.x.val.apply_binary_op(this.y.val, this.helper));
        for (let i = 0; i < this.y.val.stride.length; i++) {
            if (this.y.val.stride[i] == 0) {
                loss = loss.sum(i);
            }
        }
        this.y.backward(loss);
    }
}

class DotBackward {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
    call(loss) {
        this.x.backward(loss.dot(this.y.val.transpose(0, 1)));
        this.y.backward(this.x.val.transpose(0, 1).dot(loss));
    }
}

class PowBackward {
    constructor(x, o, e) {
        this.x = x;
        this.o = o;
        this.e = e;
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.div(this.x.val).times(this.e)));
    }
}

class MeanBackward {
    constructor(x, axis) {
        this.x = x;
        this.axis = axis;
    }
    call(loss) {
        let t_ = new Tensor(Array(this.x.val.get_dim()), this.x.val.shape);
        t_.ones();
        t_ = t_.mul(loss);
        this.x.backward(t_.times(1 / this.x.val.shape[this.axis]));
    }
}

class SigmoidBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        return x * (1 - x);
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)));
    }
}

class TanhBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        return (1 - x ** 2) / 2;
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)));
    }
}

class ReLUBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        return x > 0 ? 1 : 0;
    }
    call(loss) {
        this.x.backward(loss.mul(this.o.val.apply_unitary_op(this.helper)));
    }
}

class GELUBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        let y = ActivationFunction.sigmoid(1.702 * x);
        return y + x * 1.702 * (y * (1 - y));
    }
    call(loss) {
        this.x.backward(loss.mul(this.x.val.apply_unitary_op(this.helper)));
    }
}

class SILUBackward {
    constructor(x, o) {
        this.x = x;
        this.o = o;
    }
    helper(x) {
        let y = ActivationFunction.sigmoid(x);
        return y + x * (y * (1 - y));
    }
    call(loss) {
        this.x.backward(loss.mul(this.x.val.apply_unitary_op(this.helper)));
    }
}

// Variable class
// This a wrapper for tensor objects to be able to perform autodifferntiation
class Variable {
    constructor(t, backward_hook) {
        this.val = t;
        if (backward_hook) {
            this.grad = null;
        } else {
            this.grad = new Tensor(Array(t.data.length), [...t.shape]);
            this.grad.zeros();
        }
        this.backward_hook = backward_hook; // if this is null it means it is a leaf of the graph
    }

    backward(loss) {
        if (this.backward_hook) {
            this.backward_hook.call(loss);
        } else {
            this.grad = this.grad.add(loss);
        }
    }
    zero_grad() {
        if (this.grad) {
            this.grad.zeros();
        }
    }
    add(v) {
        let new_val = this.val.add(v.val);
        return new Variable(new_val, new AddBackward(this, v));
    }
    sub(v) {
        let new_val = this.val.sub(v.val);
        return new Variable(new_val, new SubBackward(this, v));
    }
    mul(v) {
        let new_val = this.val.mul(v.val);
        return new Variable(new_val, new MulBackward(this, v));
    }
    div(v) {
        let new_val = this.val.div(v.val);
        return new Variable(new_val, new DivBackward(this, v));
    }
    dot(v) {
        let new_val = this.val.dot(v.val);
        return new Variable(new_val, new DotBackward(this, v));
    }
    mean(axis) {
        let new_val = this.val.mean(axis);
        return new Variable(new_val, new MeanBackward(this, axis));
    }
    pow(e) {
        let new_val = this.val.pow(e);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new PowBackward(this, o, e);
        return o;
    }
    static sigmoid(v) {
        let new_val = v.val.apply_unitary_op(ActivationFunction.sigmoid);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new SigmoidBackward(v, o);
        return o;
    }
    static tanh(v) {
        let new_val = v.val.apply_unitary_op(ActivationFunction.tanh);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new TanhBackward(v, o);
        return o;
    }
    static relu(v) {
        let new_val = v.val.apply_unitary_op(ActivationFunction.relu);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new ReLUBackward(v, o);
        return o;
    }
    static gelu(v) {
        let new_val = v.val.apply_unitary_op(ActivationFunction.gelu);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new GELUBackward(v, o);
        return o;
    }
    static silu(v) {
        let new_val = v.val.apply_unitary_op(ActivationFunction.silu);
        let o = new Variable(new_val, new NopBackward());
        o.backward_hook = new SILUBackward(v, o);
        return o;
    }
}

// Loss functions
class Loss {
    static mse_loss(o, t) {
        return o.sub(t).pow(2).mean(0).mean(1);
    }

    static cross_entropy_loss() {}
}

// Optimizers

class SGD {
    constructor(params, hparams) {
        this.params = params;
        this.lr = hparams.lr;
    }

    zero_grad() {
        for (let i = 0; i < this.params.length; i++) {
            this.params[i].zero_grad();
        }
    }

    step() {
        for (let i = 0; i < this.params.length; i++) {
            this.params[i].val = this.params[i].val.sub(
                this.params[i].grad.times(this.lr)
            );
        }
    }
}

// Nerual Network Layers
class Linear {
    constructor(n_input, n_output) {
        let W_ = new Tensor(Array(n_input * n_output), [n_input, n_output]);
        let b_ = new Tensor(Array(n_output), [1, n_output]);
        W_.random();
        b_.random();
        this.W = new Variable(W_, null);
        this.b = new Variable(b_, null);
    }

    parameters() {
        return [this.W, this.b];
    }

    forward(x) {
        let o = x.dot(this.W).add(this.b);
        return o;
    }
}

class Sigmoid {
    constructor() {}

    parameters() {
        return [];
    }

    forward(x) {
        return Variable.sigmoid(x);
    }
}

class Tanh {
    constructor() {}

    parameters() {
        return [];
    }

    forward(x) {
        return Variable.sigmoid(x);
    }
}

class ReLU {
    constructor() {}

    parameters() {
        return [];
    }

    forward(x) {
        return Variable.relu(x);
    }
}

class GELU {
    constructor() {}

    parameters() {
        return [];
    }

    forward(x) {
        return Variable.gelu(x);
    }
}

class SILU {
    constructor() {}

    parameters() {
        return [];
    }

    forward(x) {
        return Variable.silu(x);
    }
}

class Sequential {
    constructor(layers) {
        this.layers = layers;
    }

    parameters() {
        let o = [];
        for (let i = 0; i < this.layers.length; i++) {
            o = [...o, ...this.layers[i].parameters()];
        }
        return o;
    }

    forward(x) {
        let o = x;
        for (let i = 0; i < this.layers.length; i++) {
            o = this.layers[i].forward(o);
        }
        return o;
    }
}

export {
    ActivationFunction,
    Tensor,
    Variable,
    NopBackward,
    Loss,
    SGD,
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    SILU,
    Sequential,
};
