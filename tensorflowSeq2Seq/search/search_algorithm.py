
import time

import tensorflow as tf

from tensorflowSeq2Seq.util.debug import my_print

class SearchAlgorithm(tf.Module):

    def __init__(self, name='search_algorithm', **kwargs):
        super(SearchAlgorithm, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.search_graph   = None
        self.EOS            = self.vocab_tgt.EOS
        self.PAD            = self.vocab_tgt.PAD
        self.V              = self.vocab_tgt.vocab_size

        self.fin_V_mask = tf.range(self.V) == self.EOS
        self.fin_V_mask = tf.reshape(self.fin_V_mask, (1, 1, self.V) )

        self.trace()

    def trace(self):

        my_print('Start tracing!')

        self.search_graph = self.search_batch.get_concrete_function(
            src=tf.TensorSpec((None, None), tf.int32)
        )

        my_print('Done tracing!')

    def search(self, search_batch_generator, output_file):

        assert self.search_graph is not None

        results = []
        step_times = []
        read_order = []
        num_sentences = 0

        start = time.perf_counter()

        for idx, (src, _, out), _ in search_batch_generator.generate_batches():

            num_sentences += src.shape[0]

            start_step = time.perf_counter()

            result = self.search_graph(src)

            end_step = time.perf_counter()
            step_times.append((end_step - start_step))

            src = self.to_string_list(src, self.vocab_src)
            result = self.to_string_list(result, self.vocab_tgt)
            out = self.to_string_list(out, self.vocab_tgt)

            self.print_search_result(src, result, out)

            results += result
            read_order += idx

        end = time.perf_counter()

        assert num_sentences == search_batch_generator.dataset.corpus_size
        
        my_print(f"Searching batch took: {end - start:4.2f}s, {(end - start) / 60:4.2f}min")
        my_print(f"Average step time: {sum(step_times)/len(step_times):4.2f}s")

        self.write_sentences_to_file_in_order(results, read_order, output_file)

        return results

    def search_tflite(self, tflite_interpreter, search_dataset):

        # Get input and output tensors.
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()

        tf_dataset = search_dataset.get_prepared_tf_dataset(self.batch_size)

        results = []
        invoking_times = []

        start = time.perf_counter()

        for src, _, out, _ in tf_dataset:

            tflite_interpreter.resize_tensor_input(0, [src.shape[0], src.shape[1]])
            tflite_interpreter.allocate_tensors()
            tflite_interpreter.set_tensor(input_details[0]['index'], src)

            start_invoke = time.perf_counter()
            tflite_interpreter.invoke()
            end_invoke = time.perf_counter()
            invoking_times.append(end_invoke - start_invoke)

            result = tflite_interpreter.get_tensor(output_details[0]['index'])

            src = self.to_string_list(src, self.vocab_src)
            result = self.to_string_list(result.tolist(), self.vocab_tgt)
            out = self.to_string_list(out, self.vocab_tgt)

            self.print_search_result(src, result, out)

            results.append(result)

        end = time.perf_counter()
        my_print(f"Search took: {end - start:4.2f}s, {(end - start) / 60:4.2f}min")
        my_print(f"Average invoking time: {sum(invoking_times)/len(invoking_times):4.2f}s")

        return results

    def search_batch(self, src):
        raise NotImplementedError()

    def to_string_list(self, inps, vocab):

        processed = []

        for hyp in inps:

            if isinstance(hyp, tf.Tensor):
                hyp = hyp.numpy().tolist()

            hyp = vocab.detokenize(hyp)
            hyp = vocab.remove_padding(hyp)

            processed.append(hyp)

        return processed

    def print_search_result(self, src, result, out):

        for s, r, o in zip(src, result, out):

            my_print('===')
            my_print('src : ', ' '.join(s[1:-1]))
            my_print('hyp : ', ' '.join(r[1:-1]))
            my_print('ref : ', ' '.join(o[:-1]))

    def write_sentences_to_file_in_order(self, results, read_order, output_file):

        with open(output_file, "w") as file:

            for i in range(len(results)):
                
                idx = read_order.index(i)
                tgt = results[idx][1:-1]
                tgt = " ".join(tgt)

                file.write(tgt + "\n")