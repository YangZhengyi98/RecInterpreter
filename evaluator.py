import re
from collections import defaultdict
from data_utils import *

def evaluate(Model, test_dataloader):

    n_total = 0
    n_recover = 0
    j = 0
    recover_dict = defaultdict(int)
    for _, sample in enumerate(test_dataloader):

        output = Model.generate_output(sample)
        j += 1

        for o, m in zip(output, sample["movie_seq"]):

            if j % 3 == 0:

                print('generated:')
                print(o)
                print('real:')
                mm = m.split('::')
                # mm = ', '.join(mm)
                # mm = 'This user has watched ' + mm + ' in the previous'
                print(mm) 
                

            n_total += 1
            n_recover_this = 0
            real_movie_list = m.split('::')
            for movie in real_movie_list:
                if re.search(movie, o):
                    n_recover += 1
                    n_recover_this += 1

            print('recover {} movies'.format(n_recover_this))
            print('**************************')
            
            recover_dict[n_recover_this] += 1

    for i in recover_dict.keys():
        recover_dict[i] = recover_dict[i] / n_total

    print('Ave Recover: {:.4f}'.format(n_recover / n_total))
    print('Detialed recover number: ', recover_dict)


if __name__ == "__main__":
        training_samples = TrainingData(data4frame())
        train_dataloader = DataLoader(training_samples, batch_size=16, shuffle=True)

        test_samples = TrainingData(data4frame_test())
        test_dataloader = DataLoader(test_samples, batch_size=32, shuffle=False)

