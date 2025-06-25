from typing import Any, Dict

from datadings.writer import Writer
from webdataset import ShardWriter


class MsgpackFileWriter(Writer):
    """
    Writer for file-based datasets.
    Requires sample dicts with a unique ``"key"`` value.
    """

    def _write_data(self, key, packed):
        if key in self._keys_set:
            raise ValueError("duplicate key %r not allowed" % key)
        self._keys.append(key)
        self._keys_set.add(key)
        self._hash.update(packed)
        self._outfile.write(packed)
        self._offsets.append(self._outfile.tell())
        self.written += 1
        # self._printer()

    def write(self, sample: Dict[str, Any]) -> int:
        """
        Write a sample to the dataset.

        Args:
            sample (Dict[str, Any]): The sample to write, must contain a unique "__key__" value.

        Returns:
            int: The number of samples written.
        """
        if "__key__" not in sample:
            raise ValueError("Sample must contain a unique '__key__' value.")
        self._write(sample["__key__"], sample)
        return self.written


class MsgpackShardWriter(ShardWriter):
    """
    Shard writer that uses MsgpackFileWriter for writing shards.
    """

    def next_stream(self) -> None:
        """
        Close the current stream and move to the next.

        Args:
            None

        Returns:
            None
        """
        self.finish()
        self.fname = self.pattern % self.shard
        if self.verbose:
            print(
                "# writing",
                self.fname,
                self.count,
                "%.1f GB" % (self.size / 1e9),
                self.total,
            )
        self.shard += 1
        if self.opener:
            self.tarstream = MsgpackFileWriter(self.opener(self.fname), **self.kw)
        else:
            self.tarstream = MsgpackFileWriter(self.fname, **self.kw)
        self.count = 0
        self.size = 0
