from .gptqutils import quantize_model


class GPTQConfig:
    """
    Configuration class for the GPTQ (Generative Pre-trained Transformer Quantization) algorithm.

    This class holds all the parameters needed to apply the GPTQ method to a model.
    Its attributes are based on the original command-line arguments from the research
    repository's `opt.py` script.

    Args:
        dataset (str): Path to the calibration dataset.
        wbits (int, optional): The number of bits to quantize the weights to.
            Defaults to 4.
        nsamples (int, optional): The number of calibration data samples to use.
            Defaults to 128.
        seqlen (int, optional): The sequence length to use for calibration.
            Defaults to 512.
        percdamp (float, optional): The percentage of Hessian damping to use.
            Defaults to 0.01.
        groupsize (int, optional): The size of the group of weights to quantize together.
            A groupsize of -1 means quantization is done per-column. Defaults to 128.
        symmetric (bool, optional): If True, uses symmetric quantization. If False,
            uses asymmetric quantization. Defaults to False.
        act_order (bool, optional): If True, quantizes columns in order of decreasing
            activation size. This can improve accuracy. Defaults to False.
    """
    def __init__(
        self,
        dataset,
        wbits: int = 4,
        nsamples: int = 128,
        seqlen: int = 512,
        percdamp: float = 0.01,
        groupsize: int = 128,
        symmetric: bool = False,
        act_order: bool = False,
    ):
        self.dataset = dataset
        self.nsamples = nsamples
        self.seqlen = seqlen
        self.percdamp = percdamp
        self.wbits = wbits
        self.groupsize = groupsize
        self.symmetric = symmetric
        self.act_order = act_order
        self.quantization_method = "gptq"  # Fixed identifier for the method

    def quantize(self, model):
        """
        Applies the GPTQ quantization to the provided model using this configuration.

        This method is a wrapper around the main `quantize_model` function from the
        gptqutils module.

        Args:
            model (keras.Model): The pre-trained Keras model to be quantized.

        Returns:
            The result of the quantization process. Note: The underlying function
            may modify the model in-place.
        """
        print("Initiating quantization from GPTQConfig...")
        return quantize_model(
            model=model,
            dataset=self.dataset,
            nsamples=self.nsamples,
            seqlen=self.seqlen,
            percdamp=self.percdamp,
            groupsize=self.groupsize,
            symmetric=self.symmetric,
            act_order=self.act_order,
            wbits=self.wbits
        )
        # Return the model, which has been modified in-place.
        return model