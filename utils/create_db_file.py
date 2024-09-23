from factscore.factscorer import FactScorer

fs = FactScorer()

# this will create a database using your file
# for English Wikipedia (18GB)), it takes ~8 hours
# once DB file is created, you can reuse it by only specifying `db_path`
ks_name = 'kowiki-20240301' # name_of_your_knowledge_source
jsonl_file_path = 'FactScore/kowiki-20240301.jsonl' # path_to_jsonl_file
db_path = '' # path_to_output_db_file
fs.register_knowledge_source(ks_name,
                             data_path=jsonl_file_path,
                             db_path=None)