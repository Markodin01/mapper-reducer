# Submit the first MapReduce job (Word Count and Filter)
hadoop jar /path/to/hadoop-streaming.jar \
    -files mapper.py,reducer.py \
    -mapper mapper.py \
    -reducer reducer.py \
    -input /path/to/input \
    -output /path/to/output1

# Prepare the filter list for the second job (assuming the reducer output from job 1 is suitable)
# This may involve a custom script to extract the word list

# Submit the second MapReduce job (Text Filtering)
hadoop jar /path/to/hadoop-streaming.jar \
    -files mapper2.py,filter_list.txt#filter_list \
    -mapper mapper2.py \
    -reducer reducer2.py \
    -input /path/to/input/original_text \
    -output /path/to/output2




hadoop jar /path/to/hadoop-streaming.jar \
    -files azure/first_stage/mapper.py,azure/first_stage/reducer.py \
    -mapper "python mapper.py" \
    -reducer "python reducer.py" \
    -input "wasbs://mapper-reducer-2024-03-27t23-41-19-695z@mapperreducerhdistorage.blob.core.windows.net/articles-data/articles_part_1.json" \
    -output "wasbs://mapper-reducer-2024-03-27t23-41-19-695z@mapperreducerhdistorage.blob.core.windows.net/articles-data/"
