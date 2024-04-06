const { Tensor, Sequential, Linear, Tanh, Variable, NopBackward, SGD, Loss } = await import("nnjs");

let model = new Sequential([
    new Linear(2, 3),
    new Tanh(),
    new Linear(3, 1)
])

let x = new Variable(new Tensor([2, 3, 5, 6, 7, 9], [3, 2]), new NopBackward())
let y = new Variable(new Tensor([3, 5, 7], [3, 1]), new NopBackward())

let optim = new SGD(model.parameters(), { lr: 0.1 });

console.log(model.forward(x).val.data)
let epochs = 10
for (let epoch = 0; epoch < epochs; epoch++) {
    optim.zero_grad();
    let o = model.forward(x);
    let l = Loss.mse_loss(o, y);
    console.log(l.val.item());
    l.backward(new Tensor([1], [1, 1]));
    optim.step();
}
console.log(model.forward(x).val.data)
