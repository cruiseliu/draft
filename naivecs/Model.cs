using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class Part : Module {
    public Module fc;
    public Module activ;

    public Part(string name, int inSize, int outSize): base(name)
    {
        fc = Linear(inSize, outSize);
        activ = ReLU();
        RegisterComponents();
    }

    public override Tensor forward(Tensor l0)
    {
        var l1 = fc.forward(l0);
        var l2 = activ.forward(l1);
        return l2;
    }
}

public class Net : Module {
    public Module part0 = new Part("part1", 20, 100);
    public Module part1 = new Part("part2", 100, 10);

    public Net(string name) : base(name)
    {
        RegisterComponents();
    }

    public override Tensor forward(Tensor l0)
    {
        var l1 = part0.forward(l0);
        var l2 = part1.forward(l1);
        return l2;
    }
}
