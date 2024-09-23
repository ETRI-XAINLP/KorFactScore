import jsonlines
import numpy as np
import sys

def comparision_with_human_judgement(decision_s, decision_h_dict):
    # decision_h_dict = {'atom':'human.verdict-all'}, {'atom':'human.verdict-all'}, ... {}
    n_afs, num_matching_decisions , n_true_human = 0, 0, 0
    n_sample = len(decision_s)
    
    n_afs = sum(1 for n in range(n_sample) if decision_h_dict.get(decision_s[n]['atom'],None) is not None)

    n_true_human = sum(decision_h_dict.get(decision_s[n]['atom']) for n in range(n_sample) if decision_h_dict.get(decision_s[n]['atom'],None) is not None)   

    n_true_system = sum(decision_s[n]['is_supported'] for n in range(n_sample))
    
    # 시스템 판단과 인간 판단이 일치하는 경우의 수 계산
    num_matching_decisions  = sum(decision_s[n]['is_supported'] is decision_h_dict.get(decision_s[n]['atom']) for n in range(n_sample) if decision_h_dict.get(decision_s[n]['atom'],None) is not None)
    
    return n_afs, n_sample, num_matching_decisions, n_true_human, n_true_system

if __name__ == "__main__": 
    if len(sys.argv) != 3:
        print("Usage: python evaluate_system_vs_human_judgements.py <human_judge_data_path> <system_decisions_data_path>")
        sys.exit(1)

    # human_judge_data = 'gen_bio-gpt_4-fr-all-gold.jsonl'
    # system_decisions_data = 'cache/gpt_4-fr-all/Meta-Llama-3.1-8B-Instruct/4_af_decisions_bm25_k5.jsonl'
    human_judge_data = sys.argv[1]
    system_decisions_data = sys.argv[2]

    dh_out = jsonlines.open(human_judge_data, mode='r')
    ds_out = jsonlines.open(system_decisions_data,mode='r')

    # human data를 dictionary로 loading
    decisions_h_dict = {}
    for d in dh_out:
        for item in d:
            decisions_h_dict[item['atom']] = item['human.verdict-all']
    # print('# decisions:', len(decisions_h_dict))
    
    n_human = 0
    total_system_score, total_human_score, total_accuracy = 0, 0, 0
    for decision_s in ds_out:
        n_afs, n_sample, num_matching_decisions , n_true_human, n_true_system = comparision_with_human_judgement(decision_s, decisions_h_dict)

        if n_afs == 0:
            n_afs = 1 # 0으로 나누는 것을 방지하기 위해
        
        total_system_score += n_true_system/n_sample
        total_human_score += n_true_human/n_afs
        total_accuracy += num_matching_decisions /n_afs
        n_human += 1
        
print('------- Final Results -----------')
print("(System 판단) 사실성 점수 = %.1f%%" % (100*(total_system_score/n_human)))
print("(Human 판단) 사실성 점수 = %.1f%%" % (100*(total_human_score/n_human)))
print("--> System의 사실판단 성능(Accuracy) = %.1f%% \n" % (100*(total_accuracy/n_human)))
