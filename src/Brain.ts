import ShortTermMemorySnapshot from "./ShortTermMemorySnapshot";
import Input from "./Input";
import Output from "./Output";

export default class Brain {
    private shortTermMemorySnapshots: ShortTermMemorySnapshot[] = [];

    readonly inputsCount: number;
    readonly outputsCount: number;

    constructor(inputsCount: number, outputsCount: number) {
        this.inputsCount = inputsCount;
        this.outputsCount = outputsCount;
    }

    public execute(inputs: Input[]): Output[]
    {
        if (inputs.length !== this.inputsCount) {
            throw new Error('Invalid inputs count');
        }

        const outputs = this.calculateOutputs(inputs);

        this.shortTermMemorySnapshots.unshift(new ShortTermMemorySnapshot(inputs, outputs));

        return outputs;
    }

    public getShortTermMemorySnapshots(): ShortTermMemorySnapshot[]
    {
        return this.shortTermMemorySnapshots;
    }

    private calculateOutputs(inputs: Input[]): Output[]
    {
        const outputs: Output[] = [];

        for (let i = 0; i < this.outputsCount; ++i) {
            outputs.push(new Output(0.0));
        }

        return outputs;
    }
}
