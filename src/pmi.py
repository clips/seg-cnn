import gzip
import json

class PMI(object):
    def __init__(self, f_pmi, dir_in='../data/', with_sem_type = True):
        """
        Initialize a dictionary with PMI values between pair of concepts
        :param f_pmi: File name containing PMI values
        :param dir_in: directory containing f_pmi
        :param with_sem_type: True if semantic types are paired with concept mentions in external file
        """
        with gzip.GzipFile(dir_in + f_pmi, 'r') as f:
            self.pmi_dict = json.loads(f.read().decode('utf-8'), encoding="utf-8")
            self.with_sem_type = with_sem_type

    def get_pmi(self, c1, c2, c1_type, c2_type):
        """
        :param c1: concept 1
        :param c2: concept 2
        :param c1_type: semanatic type of concept1
        :param c2_type: semantic type of concept2
        :return: pmi score between c1 and c2, matched with precomputed file from external corpus
        """

        if type(c1) == list:
            c1 = '_'.join(c1)
        if type(c2) == list:
            c2 = '_'.join(c2)

        if self.with_sem_type:
            c1 = (c1, c1_type)
            c2 = (c2, c2_type)

        pmi = self._get_symmetric_pmi(c1, c2)

        if pmi is None:
            print("Getting backoff PMI between {} and {}".format(c1, c2))
            pmi = self._get_backoff_pmi(c1, c2, 1)
            if pmi is None:
                # pmi = float("inf") #pmi not found
                pmi = 0.
                # print("No matching entry found for {} and {}".format(c1, c2))

        return pmi

    def _get_symmetric_pmi(self, c1, c2):
        """
        Return the PMI score of (c1, c2) or (c2, c1)
        :param c1: concept1
        :param c2: concept2
        :return: pmi value if pair found in external corpus, None otherwise
        """

        if repr((c1, c2)) in self.pmi_dict:
            return self.pmi_dict[repr((c1, c2))]
        elif repr((c2, c1)) in self.pmi_dict:
            return self.pmi_dict[repr((c2, c1))]
        else:
            return None

    def _get_backoff_pmi(self, c1, c2, cur_depth, with_max_depth=True, max_depth=2):
        """
        Get the PMI match by backing off the starting terms of the query concept
        :param c1: concept1
        :param c2: concept2
        :param cur_depth: Current level of backoff
        :param with_max_depth: True to limit the number of terms to remove for getting a match in external corpus
        :param max_depth: maximum number of query terms to back off
        :return: PMI value if a match is found, None otherwise
        """

        pmi = self._backoff_one_term(c1, c2, cur_depth, with_max_depth=with_max_depth,
                                max_depth=max_depth)
        if pmi:
            return pmi

        pmi = self._backoff_one_term(c2, c1, cur_depth, with_max_depth=with_max_depth,
                                max_depth=max_depth)

        return pmi

    def _backoff_one_term(self, backoff_con, con, cur_depth, with_max_depth, max_depth):

        if self.with_sem_type:
            backoff_term = backoff_con[0]
            term = con[0]
        else:
            backoff_term = backoff_con
            term = con

        while '_' in backoff_term:
            backoff_term = backoff_term[backoff_term.index('_') + 1:]
            if self.with_sem_type:
                backoff_term = (backoff_term, backoff_con[1])
                term = (term, con[1])

            pmi = self._get_symmetric_pmi(backoff_term, term)

            if pmi:
                print("Got PMI between {} and {}".format(backoff_term, term))
                return pmi
            elif (not with_max_depth) or (with_max_depth and cur_depth <= max_depth):
                return self._get_backoff_pmi(backoff_term, term, cur_depth + 1)

        return None

