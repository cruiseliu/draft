using System;
using System.Diagnostics;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

Tensor ReadTensor(string fileName, long[] shape, string dtypeName)
{
    long n = shape.Aggregate((long)1, (x, y) => x * y);
    var bytes = File.ReadAllBytes("weights/" + fileName);
    Tensor? tensor = null;

    if (dtypeName == "torch.float32") {
        var data = new float[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.float32);

    } else if (dtypeName == "torch.float64") {
        var data = new double[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.float32);

    } else if (dtypeName == "torch.uint8") {
        tensor = torch.tensor(bytes, torch.uint8);

    } else if (dtypeName == "torch.int8") {
        var data = new sbyte[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.int8);

    } else if (dtypeName == "torch.int16") {
        var data = new short[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.int16);

    } else if (dtypeName == "torch.int32") {
        var data = new int[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.int32);

    } else if (dtypeName == "torch.int64") {
        var data = new long[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.int64);

    } else if (dtypeName == "torch.bool") {
        var data = new bool[n];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        tensor = torch.tensor(data, torch.@bool);
    }

    return tensor!.reshape(shape);
}

void SetWeight(Module model, string layerName, Tensor tensor)
{
    object current = model;
    var parts = layerName.Split('.');
    foreach (var part in parts.SkipLast(1)) {
        current = current.GetType()!.GetField(part)!.GetValue(current)!;
    }
    current.GetType()!.GetProperty(parts.Last())!.SetValue(current, tensor);
}

void LoadWeights(Module model)
{
    var metaJson = File.ReadAllText("weights_meta.json");
    var arr = JsonConvert.DeserializeObject<JArray>(metaJson);
    foreach (dynamic obj in arr!) {
        var shape = ((JArray)obj.shape).Select(x => (long)x).ToArray();
        var tensor = ReadTensor(obj.name, shape, obj.dtype);
        SetWeight(model, obj.cs_name, tensor);
    }
}

Console.WriteLine("Hello, World!");

var model = new Net("naive");

float[] xArray = new float[] {
    0.2041F, 0.2063F, 0.2980F, 0.5355F, 0.6694F,
    0.2127F, 0.4529F, 0.4136F, 0.9562F, 0.3197F,
    0.9845F, 0.1933F, 0.2940F, 0.1239F, 0.8684F,
    0.1562F, 0.7070F, 0.3163F, 0.9983F, 0.5705F
};

var x = torch.tensor(xArray);

Console.WriteLine(x);

LoadWeights(model);

var y = model.forward(x);

for (var i = 0; i < 10; i++) {
    Console.WriteLine(y.ReadCpuSingle(i));
}
