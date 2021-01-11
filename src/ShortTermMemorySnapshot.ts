import Input from "./Input";
import Output from "./Output";

export default class ShortTermMemorySnapshot {
    readonly inputs: Input[];
    readonly outputs: Output[];

    constructor(inputs: Input[], outputs: Output[]) {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}
