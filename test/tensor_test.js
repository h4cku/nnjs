import { Tensor } from "nnjs";

let x = new Tensor([8, 2, 3, 4, 8, 1, 9, 0], [4, 2])
console.log(Tensor.argmax(x, 1));