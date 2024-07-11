from transformers.cache_utils import StaticCache


class ParlerTTStaicCache(StaticCache):
    def __getitem__(self, layer_idx: int):
        return (
            self.key_cache[layer_idx], 
            self.value_cache[layer_idx]
        )
