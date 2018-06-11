import gzip
from collections import defaultdict

class PMI(object):
    def __init__(self, f_pmi, dir_in='../data/', with_sem_type = True):
        """
        Initialize a dictionary with PMI values between pair of concepts
        :param f_pmi: File name containing PMI values.
                      Contains tab separated values: c1 c1_type c2 c2_type pmi, starting with header line
        :param dir_in: directory containing f_pmi
        :param with_sem_type: True if semantic types are paired with concept mentions in external file
        """

        self.with_sem_type = with_sem_type

        print("Loading PMI dictionary...")

        self.pmi_dict = defaultdict(lambda: defaultdict(float)) # {term1:{term2: pmi}}
        with gzip.open(dir_in + f_pmi, 'r') as f:
            for i, line in enumerate(f):
                if i == 0: #header line
                    continue

                line = line.split('\t') #tab separated file

                if (self.with_sem_type and len(line) != 5) or (not self.with_sem_type and len(line) != 3):
                    print("Error in line: ", line)
                    continue

                if self.with_sem_type:
                    term1 = (line[0], line[1])
                    term2 = (line[2], line[3])
                else:
                    term1 = (line[0])
                    term2 = (line[1])

                pmi = float(line[-1])

                self.pmi_dict[term1][term2] = pmi

        self.pmi_dict.default_factory = None

        print("Done")



    def get_pmi(self, c1, c2, c1_type, c2_type):
        """
        :param c1: concept 1
        :param c2: concept 2
        :param c1_type: semantic type of concept1
        :param c2_type: semantic type of concept2
        :return: pmi score between c1 and c2, matched with precomputed file from external corpus
        """

        if type(c1) == list:
            c1 = '_'.join(c1)
        if type(c2) == list:
            c2 = '_'.join(c2)

        pmi = self._get_symmetric_pmi(c1, c2, c1_type, c2_type)

        if pmi is None:
            print("Getting backoff PMI between {} and {}".format(c1, c2))
            pmi = self._get_backoff_pmi(c1, c2, c1_type, c2_type, 1)
            if pmi is None:
                pmi = 0.

        print("Got PMI value of {} between {} and {}".format(pmi, c1, c2))

        # pmi = self.discretize_pmi(pmi)

        return pmi

    def discretize_pmi(self, pmi):
        if pmi < 0.:
            pmi = -100.
        elif pmi > 0.:
            pmi = 100.

        # print(pmi)
        return pmi

    def _get_symmetric_pmi(self, c1, c2, c1_type, c2_type):
        """
        Return the PMI score of (c1, c2) or (c2, c1)
        :param c1: concept1
        :param c2: concept2
        :return: pmi value if pair found in external corpus, None otherwise
        """
        pmi = self._get_pmi_for_pair(c1, c2, c1_type, c2_type)
        if pmi is None:
            pmi = self._get_pmi_for_pair(c2, c1, c2_type, c1_type)

        if pmi:
            return pmi
        else:
            return None

    def _get_pmi_for_pair(self, c1, c2, c1_type, c2_type):

        if self.with_sem_type:
            term1 = (c1, c1_type)
            term2 = (c2, c2_type)
        else:
            term1 = (c1)
            term2 = (c2)
        try:
            return self.pmi_dict[term1][term2]
        except KeyError:
            return None

    def _get_backoff_pmi(self, c1, c2, c1_type, c2_type, cur_depth, with_max_depth=True, max_depth=2):
        """
        Get the PMI match by backing off the starting terms of the query concept
        :param c1: concept1
        :param c2: concept2
        :param c1_type: semantic type of concept1
        :param c2_type: semantic type of concept2
        :param cur_depth: Current level of backoff
        :param with_max_depth: True to limit the number of terms to remove for getting a match in external corpus
        :param max_depth: maximum number of query terms to back off
        :return: PMI value if a match is found, None otherwise
        """

        pmi = self._backoff_one_term(c1, c2, c1_type, c2_type, cur_depth, with_max_depth=with_max_depth,
                                max_depth=max_depth)
        if pmi:
            return pmi

        pmi = self._backoff_one_term(c2, c1, c2_type, c1_type, cur_depth, with_max_depth=with_max_depth,
                                max_depth=max_depth)

        return pmi

    def _backoff_one_term(self, backoff_con, con, backoff_type, con_type, cur_depth, with_max_depth, max_depth):

        backoff_term = backoff_con
        term = con

        while '_' in backoff_term:
            backoff_term = backoff_term[backoff_term.index('_') + 1:]
            pmi = self._get_symmetric_pmi(backoff_term, term, backoff_type, con_type)

            if pmi:
                return pmi
            elif (not with_max_depth) or (with_max_depth and cur_depth <= max_depth):
                return self._get_backoff_pmi(backoff_term, term, backoff_type, con_type, cur_depth + 1)

        return None