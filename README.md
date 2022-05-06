# TraceSim: An Alignment Method for Computing Stack Trace Similarity

By Irving Muller Rodrigues, Aleksandr Khvorov, Daniel Aloise, Roman Vasiliev, Dmitrij Koznov, Eraldo Rezende Fernandes, George Chernishev, Dmitry Luciv,
Nikita Povarov

## Abstract


Software systems can automatically submit crash reports to a repository for investigation when program failures occur. A significant portion of these crash reports are duplicate, i.e., they are caused by the same software issue. Therefore, if the volume of submitted reports is very large, automatic grouping of duplicate crash reports can significantly ease and speed up analysis of software failures. This task is known as crash report deduplication. Given a huge volume of incoming reports, increasing quality of deduplication is an important task. The majority of studies address it via information retrieval or sequence matching methods based on the similarity of stack traces from two crash reports. While information retrieval methods disregard the position of a frame in a stack trace, the existing works based on sequence matching algorithms do not fully consider subroutine global frequency and unmatched frames. Besides, due to data distribution differences among software projects, parameters that are learned using machine learning algorithms are necessary to provide more flexibility to the methods. In this paper, we propose TraceSim – an approach for crash report deduplication which combines TF-IDF, optimum global alignment, and machine learning (ML)
in a novel way. Moreover, we propose a new evaluation methodology for this task that is more comprehensive and robust than previously used evaluation approaches. TraceSim significantly outperforms seven baselines and state-of-the-art methods in the majority of the scenarios. It is the only approach that achieves competitive results on all datasets regarding all considered metrics. Moreover, we conduct an extensive ablation study that demonstrates the importance of each TraceSim’s element to its final performance and robustness. Finally, we provide the source code for all considered methods and evaluation methodology as well
as the created datasets.

## Install

Install the following packages:

```bash
conda install -c conda-forge hyperopt
conda install -c anaconda scikit-learn
conda install numpy
conda install -c conda-forge cython
conda install -c anaconda nltk
conda install -c anaconda gensim

python setup.py build_ext --inplace
```


## Data

The data used in the paper can be found [here](https://zenodo.org/record/5746044#.YabKILtyZH4). The four folders contain the dataset of the open-sources projects (Ubuntu, Eclipse, Netbeans, and Gnome).
The original data from Ubuntu can be found on [Campbell's work](https://ieeexplore.ieee.org/document/7832906).


## TF-IDF

We use the Lucene's implementation of TF-IDF. The Java code to run the experiments can be found on _textual_similarity_deduplication.zip_. 
    
## Usage

Below, we present example of how to run each method in a specific chunk:

```
# TraceSim
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  trace_sim  $WORKSPACE/TraceSim_EMSE/space_script/trace_sim_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt

# Moroo
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  moroo  $WORKSPACE/TraceSim_EMSE/space_script/mooro_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt  -top_n_file_validation $WORKSPACE/validation_result_files/lerch_eclipse_validation_0.sparse -top_n_file_test $WORKSPACE/test_result_files/lerch_eclipse_test_0.sparse

# PDM
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  pdm  $WORKSPACE/TraceSim_EMSE/space_script/pdm_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt


#Prefix Match
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  prefix_match  $WORKSPACE/TraceSim_EMSE/space_script/prefix_match_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt

# Brodie
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  brodie_05  $WORKSPACE/TraceSim_EMSE/space_script/brodie_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt

# NW
python experiments/hyperparameter_opt.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  opt_align $WORKSPACE/TraceSim_EMSE/space_script/opt_align_space_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -max_evals 100 -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt

# DURFEX
python experiments/grid_search.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt  durfex $WORKSPACE/TraceSim_EMSE/space_script/durfex_eclipse.py -nthreads 20 -filter_func threshold_trim -sparse -w 730 -test $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt


#TF-IDF
   cd $WORKSPACE/textual_similarity_deduplication
   /usr/lib/jvm/java-1.11.0-openjdk-amd64/bin/java -Dfile.encoding=UTF-8 -classpath $WORKSPACE/textual_similarity_deduplication/out/production/textual_similarity_deduplication:$WORKSPACE/textual_similarity_deduplication/lib/json-simple-1.1.1.jar:$WORKSPACE/textual_similarity_deduplication/lib/argparse4j-0.8.1.jar:$WORKSPACE/textual_similarity_deduplication/lib/commons-io-2.7.jar:$WORKSPACE/textual_similarity_deduplication/lib/commons-lang3-3.10.jar:$WORKSPACE/textual_similarity_deduplication/lib/lucene-analyzers-common-8.5.2.jar:$WORKSPACE/textual_similarity_deduplication/lib/lucene-core-8.5.2.jar:$WORKSPACE/textual_similarity_deduplication/lib/lucene-queryparser-8.5.2.jar:/home/irving/ideaIU-2020.1.2/idea-IU-201.7846.76/lib/idea_rt.jar BugDeduplication -db $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json -training $DATASET_PATH/eclipse_2018/chunks_test/training_chunk_0.txt $DATASET_PATH/eclipse_2018/chunks_test/validation_chunk_0.txt -validation $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt  -index /home/irving/lucene_index -out $WORKSPACE/textual_similarity_deduplication/test_result_files/lerch_eclipse_test_0.sparse -config $WORKSPACE/textual_similarity_deduplication/test_config_eclipse.json -sparse -min_top_score 500 -window 730 > $WORKSPACE/TraceSim_EMSE/lerch_eclipse_test_results/lerch_eclipse_test_java_run_0.log
   cd  $WORKSPACE/TraceSim_EMSE
   python experiments/calculate_metrics.py $DATASET_PATH/eclipse_2018/eclipse_stacktraces.json $DATASET_PATH/eclipse_2018/chunks_test/test_chunk_0.txt -w 730 -add_cand -result_file $WORKSPACE/textual_similarity_deduplication/test_result_files/lerch_eclipse_test_0.sparse 2> $WORKSPACE/TraceSim_EMSE/lerch_eclipse_test_results/lerch_eclipse_test_python_run_0.log


```

# Results

Paper results are in the following jupyter notebooks: _test_results.ipynb_ and _ablation_study_results.ipynb_.

# Citation
The paper was accepted and will be published in EMSE - Topical Collection: Machine Learning Techniques for Software Quality Evaluation (MaLTeSQuE).

@Article{rodrigues2021,
    author={Rodrigues, Irving Muller
        and Khvorov, Aleksandr
        and Aloise, Daniel
        and Vasiliev, Roman
        and Koznov, Dmitrij
        and Fernandes, Eraldo Rezende
        and Chernishev, George
        and Luciv, Dmitry
        and Povarov, Nikita},
    title={TraceSim: An Alignment Method for Computing Stack Trace Similarity},
    journal={Empirical Software Engineering},
    year={2022},
    month={Mar},
    day={01},
    volume={27},
    number={2},
    pages={53},
    issn={1573-7616},
    doi={10.1007/s10664-021-10070-w},
    url={https://doi.org/10.1007/s10664-021-10070-w},
    publisher={Springer}
}

# License
```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
