from numpy import array

from numpy import argmax
from math import log

# beam search

# def beam_search_decoder(data, k):
#
#    sequences = [[list(), 0.0]]
#
#    # walk over each step in sequence
#
#    for row in data:
#
#        all_candidates = list()
#
#        # expand each current candidate
#
#        for i in range(len(sequences)):
#
#            seq, score = sequences[i]
#
#            for j in range(len(row)):
#
#                candidate = [seq + [j], score + (-log(row[j]))]
#
#                all_candidates.append(candidate)
#
#        # order all candidates by score
#
#        ordered = sorted(all_candidates, key=lambda tup:tup[1])
#
#        # select k best
#
#        sequences = ordered[:k]
#
#    return sequences


def beam_search_decoder(data, k = 10):

   sequences = [[list(), 0.0]]

   # walk over each step in sequence

   for row in data:

       all_candidates = list()

       # expand each current candidate

       for i in range(len(sequences)):

           seq, score = sequences[i]

           for j in range(len(row)):
               seq_ = seq.copy()
               if seq and seq[-1] == j and j!=20 and seq[-1]!=20:
                   if seq[-1] == 20:
                       seq_.remove(20)
                   candidate = [seq_, score + (-log(row[j]))]
               else:
                   if seq and seq[-1] == 20:
                       seq_.remove(20)
                   candidate = [seq_ + [j], score + (-log(row[j]))]



               all_candidates.append(candidate)

       # order all candidates by score

       ordered = sorted(all_candidates, key=lambda tup:tup[1])

       # select k best

       # sequences = ordered[:k]

       sequence_list = []
       sequence_temp = []
       for i,sequence in enumerate(ordered):
           if i == 0:
               sequence_list.append(sequence)
               sequence_temp.append(sequence[0])
           else:
               if sequence[0] not in sequence_temp:
                   sequence_list.append(sequence)
                   sequence_temp.append(sequence[0])
           if len(sequence_list) == k:
               break
       sequences = sequence_list

   return sequences

if __name__ == '__main__':


    # define a sequence of 10 words over a vocab of 5 words

    data = [[0.1, 0.2, 0.3, 0.4, 0.5],

           [0.5, 0.4, 0.3, 0.2, 0.1],

           [0.1, 0.2, 0.3, 0.4, 0.5],

           [0.5, 0.4, 0.3, 0.2, 0.1],

           [0.1, 0.2, 0.3, 0.4, 0.5],

           [0.5, 0.4, 0.3, 0.2, 0.1],

           [0.1, 0.2, 0.3, 0.4, 0.5],

           [0.5, 0.4, 0.3, 0.2, 0.1],

           [0.1, 0.2, 0.3, 0.4, 0.5],

           [0.5, 0.4, 0.3, 0.2, 0.1]]

    data = array(data)

    # decode sequence

    result = beam_search_decoder(data, 3)

    # print result

    for seq in result:

       print(seq)