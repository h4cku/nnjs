// const { Tensor, Sequential, Linear, Tanh, Variable, NopBackward, SGD, Loss } = await import("nnjs");
import {
    ReLU,
    Tensor,
    Sequential,
    Linear,
    Tanh,
    Variable,
    NopBackward,
    SGD,
    Loss,
    GELU,
    SILU,
    Sigmoid,
    Softmax,
} from "nnjs";

let model = new Sequential([
    new Linear(2, 5),
    new Sigmoid(),
    new Linear(5, 5),
    new Sigmoid(),
    new Linear(5, 2),
    new Softmax(),
]);

let x = new Variable(new Tensor([2, 3, 5, 6, 7, 9], [3, 2]), new NopBackward());
let y = new Variable(new Tensor([1, 0, 0, 1, 0, 1], [3, 2]), new NopBackward());

let optim = new SGD(model.parameters(), { lr: 1 });

console.log(model.forward(x).val.data);
let epochs = 100;
for (let epoch = 0; epoch < epochs; epoch++) {
    optim.zero_grad();
    let o = model.forward(x);
    let l = Loss.cross_entropy_loss(o, y);
    console.log(l.val.item());
    l.backward(new Tensor([1], [1, 1]));
    optim.step();
}
console.log(model.forward(x).val.data);
