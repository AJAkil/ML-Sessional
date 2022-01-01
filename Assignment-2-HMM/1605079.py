import numpy as np


class HMM:
    def __init__(self, transition_probs, gaussian_params, num_states, observations):
        self.transition_probs = transition_probs
        self.gaussian_params = gaussian_params
        self.num_states = num_states
        self.observations = observations
        self.initial_state = []

    def show_hmm_params(self):
        print(f'Number of States {self.num_states}')
        print(f'Transition Probabilities:\n {self.transition_probs}')
        print(f'Gaussian Params\n {self.gaussian_params}')

    def set_initial_probs(self):
        # modifying the transition probability matrices by subtracting diagonal elements
        transition_probs = self.transition_probs.T
        modified_diag_elems = np.diagonal(transition_probs) - 1
        row, col = np.diag_indices(transition_probs.shape[0])
        transition_probs[row, col] = modified_diag_elems
        # print(transition_probs)

        # taking all rows except the last one
        transition_probs = transition_probs[:-1, :]

        # print(transition_probs.shape)

        # creating the coefficient matrix
        coefficient_matrix = np.vstack(
            (transition_probs,
             np.ones(transition_probs.shape))
        )

        # creating the vector on left side
        b = np.zeros(coefficient_matrix.shape[0])
        b[-1] = 1

        print(coefficient_matrix)
        print(b)

        # solving to get the initial state
        self.initial_state = np.linalg.solve(coefficient_matrix, b)

    def viterbi(self):
        total_time_stamps = len(self.observations)
        """
        we set up two matrices to keep track. One will store the probability of
        each state at each time stamp and the other will keep track of the indices
        of the backtracking values
        """
        probability_matrix = np.zeros((self.num_states, total_time_stamps))

    def baulm_welch_learn(self):
        pass


class FileHandler:
    pass


if __name__ == '__main__':

    observations = open('./Sample input and output for HMM/Input/data.txt', 'r').readlines()
    parameters = open('./Sample input and output for HMM/Input/parameters.txt.txt', 'r').readlines()

    parameters = [params.strip() for params in parameters]
    observations = [float(y.strip()) for y in observations]

    num_states = int(parameters[0])
    transition_probs = []
    gaussian_params = []

    # extracting the transition probabilities
    for index in range(1, num_states + 1):
        temp = [float(trans_prob) for trans_prob in parameters[index].split()]
        transition_probs.append(temp)

    # extracting the gaussian parameters
    for index in range(num_states + 1, len(parameters)):
        temp = [float(params) for params in parameters[index].split()]
        gaussian_params.append(temp)

    hmm = HMM(
        transition_probs=np.array(transition_probs),
        gaussian_params=np.array(gaussian_params),
        num_states=num_states,
        observations=np.array(observations)
    )

    # hmm.show_hmm_params()
    hmm.set_initial_probs()
    # print(hmm.observations)
