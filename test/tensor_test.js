import { Tensor } from "nnjs";

let x;

// One hot encoding
x = Tensor.one_hot_encoding([1, 2, 4, 3], 5);
console.log(x);

// Argmax
x = new Tensor([8, 2, 3, 4, 8, 1, 9, 0], [4, 2]);
console.log(Tensor.argmax(x, 0));
