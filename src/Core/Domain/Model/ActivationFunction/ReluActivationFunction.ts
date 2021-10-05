import ActivationFunction from "./ActivationFunction";

export default class ReluActivationFunction extends ActivationFunction
{
    process(value: number): number {
        return value < 0.0 ? 0.0 : value;
    }
}
