import {
    Tensor,
    Sequential,
    Linear,
    Variable,
    NopBackward,
    SGD,
    Adam,
    Loss,
    SILU,
    Softmax,
    ReLU,
} from "nnjs";

import * as fs from "fs";
import * as readline from "readline";

// Reading the file containing the data
const readStream = fs.createReadStream("./datasets/digits.csv");
const readInterface = readline.createInterface({
    input: readStream,
});
const output = [];
readInterface.on("line", (line) => {
    const row = line.split(",");
    row.forEach((e, idx, arr) => {
        row[idx] = Number(e);
    });
    output.push(row);
});

readInterface.on("close", () => {
    // After finishing reading the file we start training our model
    let model = new Sequential([
        new Linear(64, 32),
        new ReLU(),
        new Linear(32, 32),
        new ReLU(),
        new Linear(32, 10),
        new Softmax(),
    ]);

    let xs = [];
    let ys = [];

    let batch_size = 128;
    let cur = 0;
    for (let i = 0; i < Math.ceil(output.length / batch_size); i++) {
        let n = output.length - cur;
        n = n > batch_size ? batch_size : n;
        let tmpX = [];
        let tmpy = [];
        for (let j = 0; j < n; j++) {
            let row = output[cur + j];
            tmpy.push(row.pop());
            tmpX = [...tmpX, ...row];
        }
        xs.push(
            new Variable(
                new Tensor(tmpX, [n, 64]).times(1 / 16),
                new NopBackward()
            )
        );
        ys.push(
            new Variable(Tensor.one_hot_encoding(tmpy, 10), new NopBackward())
        );
        cur += n;
    }

    let optim = new Adam(model.parameters(), { lr: 0.01 });
    let epochs = 10;
    for (let epoch = 0; epoch < epochs; epoch++) {
        let tot_loss = 0;
        for (let i = 0; i < xs.length; i++) {
            optim.zero_grad();
            let o = model.forward(xs[i]);
            let l = Loss.cross_entropy_loss(o, ys[i]);
            tot_loss += l.val.item();
            l.backward(new Tensor([1], [1, 1]));
            optim.step();
        }
        console.log(tot_loss);
    }
    let o = model.forward(xs[0]);
    o = Tensor.argmax(o.val, 1);
    console.log(o);
    console.log(Tensor.argmax(ys[0].val, 1));
});

readInterface.on("error", (err) => {
    console.error("Error: ", err);
});
