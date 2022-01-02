import numpy as np
from statistics import NormalDist

np.random.seed(79)


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
        print(f'Transition Probabilities Shape: {self.transition_probs.shape}')
        print(f'Gaussian Params\n {self.gaussian_params}')
        print(f'Gaussian Params Shape:\n {self.gaussian_params.shape}')
        print(f'Observation Params Shape:\n {self.observations.shape}')

    def set_initial_probs(self):
        # modifying the transition probability matrices by subtracting diagonal elements
        transition_probs = np.copy(self.transition_probs).T
        modified_diag_elems = np.diagonal(transition_probs) - 1
        row, col = np.diag_indices(transition_probs.shape[0])
        transition_probs[row, col] = modified_diag_elems

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

        # print(coefficient_matrix)
        # print(b)

        # solving to get the initial state
        self.initial_state = np.linalg.solve(coefficient_matrix, b)

    def run_viterbi(self):
        total_time_stamps = len(self.observations)
        """
        we set up two matrices to keep track. One will store the probability of
        each state at each time stamp and the other will keep track of the indices
        of the backtracking values
        """
        probability_matrix = np.zeros((self.num_states, total_time_stamps))
        max_state_index_tracker = np.zeros((self.num_states, total_time_stamps)).astype(np.int32)

        assert probability_matrix.shape == (self.num_states, total_time_stamps)
        assert max_state_index_tracker.shape == (self.num_states, total_time_stamps)

        # setting the initial state of the probability matrix and the max state tracker matrix
        for state_no in range(self.num_states):
            # print('Setting initial probabilites')
            # print(self.observations[0])
            # # setting the first column(s) of probability matrix
            # print(self.initial_state[state_no])
            # print(self._get_emission_prob(state_no=state_no, time_stamp=0))
            #
            # print('Start:',
            #       np.log(self.initial_state[state_no] * self._get_emission_prob(state_no=state_no, time_stamp=0)))
            probability_matrix[state_no, 0] = np.log(self.initial_state[state_no] *
                                                     self._get_emission_prob(state_no=state_no, time_stamp=0))

        max_state_index_tracker[:, 0] = 0  # as the first state did not emerge from other state
        # print(probability_matrix)

        for time_stamp in range(1, total_time_stamps):
            for state_no in range(self.num_states):
                # first calculate the emission probability from gaussian pdf
                emission_current_state = self._get_emission_prob(state_no=state_no, time_stamp=time_stamp)

                """
                for state a:
                    for updating a2 we calculate the following
                    temp = P[a1, b1] + log(T[a1->a2, b1->a2] * e_a2]
                    temp (2,1)
                    P[a2] = np.max(temp [something, something])
                """
                # print("DEBUG")
                # print(probability_matrix[:, time_stamp - 1].reshape(num_states, 1))
                # print(emission_current_state)
                # print(np.log(self.transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state))

                # calculate temp value by slicing
                temp = probability_matrix[:, time_stamp - 1].reshape(num_states, 1) \
                       + np.log(self.transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state)

                assert temp.shape == (num_states, 1)

                # setting the max of all the states to update the value of a state in the current time stamp
                probability_matrix[state_no, time_stamp] = np.max(temp)
                # print(np.max(temp))

                # find argmax and set the max state index tracker matrix ( saved from t1 )
                max_state_index_tracker[state_no, time_stamp] = np.argmax(temp)

        # Backtrack the values
        hidden_states = np.zeros(total_time_stamps).astype(np.int32)
        hidden_states[-1] = np.argmax(probability_matrix[:, -1])

        for time_stamp in reversed(range(1, total_time_stamps)):
            hidden_states[time_stamp - 1] = max_state_index_tracker[hidden_states[time_stamp], time_stamp]

        return hidden_states, probability_matrix, max_state_index_tracker

    def generate_most_probable_states(self):
        hidden_states, probability_matrix, state_index = self.run_viterbi()
        print(len(hidden_states))
        most_probable_states = ["El Nino\n" if state == 1 else "La Nina\n" for state in hidden_states]

        output = open(
            '/home/akil/Work/Work/Academics/4-2/ML/Assignment-2-HMM/Sample input and output for HMM/Output/output.txt',
            'w+')
        probaility = open(
            '/home/akil/Work/Work/Academics/4-2/ML/Assignment-2-HMM/Sample input and output for HMM/Output/probabilities.txt',
            'w+')

        state = open(
            '/home/akil/Work/Work/Academics/4-2/ML/Assignment-2-HMM/Sample input and output for HMM/Output/state.txt',
            'w+')

        output.writelines(most_probable_states)

        for col in range(probability_matrix.shape[1]):
            data = ' '.join([str(val) for val in probability_matrix[:, col]])
            probaility.write(data + '\n')

        for col in range(state_index.shape[1]):
            data = ' '.join([str(val) for val in state_index[:, col]])
            state.write(data + '\n')

    def _get_emission_prob(self, state_no, time_stamp):
        prob_dist_params = self.gaussian_params[:, state_no]
        return NormalDist(mu=prob_dist_params[0], sigma=prob_dist_params[1]). \
            pdf(self.observations[time_stamp])

    def _get_emission_prob_EM(self, state_no, time_stamp, gaussian_params):
        # print('In here', gaussian_params, time_stamp, state_no)
        prob_dist_params = gaussian_params[:, state_no]
        return NormalDist(mu=prob_dist_params[0], sigma=prob_dist_params[1]). \
            pdf(self.observations[time_stamp])

    def calculate_forward_probs(self, transition_probs, gaussian_parameters):
        total_time_stamps = len(self.observations)
        """
        we set up one matrix to keep track of the outcome likelihood
        """
        forward_matrix = np.zeros((self.num_states, total_time_stamps), dtype=np.float64)

        # setting the initial state of the forward matrix
        for state_no in range(self.num_states):
            forward_matrix[state_no, 0] = self.initial_state[state_no] * \
                                          self._get_emission_prob_EM(state_no=state_no, time_stamp=0,
                                                                     gaussian_params=gaussian_parameters)

        # normalize to avoid division by zero error
        forward_matrix[:, 0] = forward_matrix[:, 0] / np.sum(forward_matrix[:, 0])

        for time_stamp in range(1, total_time_stamps):
            for state_no in range(self.num_states):
                # first calculate the emission probability from gaussian pdf
                emission_current_state = self._get_emission_prob_EM(state_no=state_no, time_stamp=time_stamp,
                                                                    gaussian_params=gaussian_parameters)

                """
                for state a:
                    for updating a2 we calculate the following
                    temp = f[a1, b1] * T[a1->a2, b1->a2] * e_a2 (element wise product)
                    temp (2,1)
                    f[a2] = np.sum(temp [something, something], axis=column wise)
                """

                a = forward_matrix[:, time_stamp - 1]
                b = transition_probs[:, state_no]
                c = emission_current_state

                # calculate temp value by slicing
                temp = forward_matrix[:, time_stamp - 1].reshape(num_states, 1) * \
                       transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state

                assert temp.shape == (num_states, 1)

                p = np.sum(temp, axis=0)

                # summing along the axes of the temp vector
                forward_matrix[state_no, time_stamp] = np.sum(temp, axis=0)

            # normalize to avoid division by zero error
            forward_matrix[:, time_stamp] = forward_matrix[:, time_stamp] / np.sum(forward_matrix[:, time_stamp])
            # print(forward_matrix)

            assert forward_matrix.shape == (num_states, total_time_stamps)

        # print(forward_matrix[:, 990:])
        f_sink = np.sum(forward_matrix[:, -1])

        return forward_matrix, f_sink

    def calculate_backward_probs(self, transition_probs, gaussian_parameters):
        total_time_stamps = len(self.observations)
        """
        we set up one matrix to keep track of the outcome likelihood
        """
        backward_matrix = np.zeros((self.num_states, total_time_stamps), dtype=np.float64)

        # setting the initial state of the backward matrix's last column
        for state_no in range(self.num_states):
            backward_matrix[state_no, -1] = 1

        backward_matrix[:, -1] = backward_matrix[:, -1] / np.sum(backward_matrix[:, -1])

        for time_stamp in reversed(range(1, total_time_stamps)):
            for state_no in range(self.num_states):
                # first calculate the emission probability from gaussian pdf
                emission_current_state = self._get_emission_prob_EM(state_no=state_no, time_stamp=time_stamp,
                                                                    gaussian_params=gaussian_parameters)

                """
                for state a:
                    for updating a2 we calculate the following
                    temp = b[a3, b3] * T[a3->a2, b3->a2] * e_a3 (element wise product)
                    temp (2,1)
                    f[a2] = np.sum(temp [something, something], axis=column wise)
                """

                # calculate temp value by slicing
                temp = backward_matrix[:, time_stamp].reshape(num_states, 1) * \
                       transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state

                # print(time_stamp, state_no, temp)

                assert temp.shape == (num_states, 1)

                # summing along the axes of the temp vector
                backward_matrix[state_no, time_stamp - 1] = np.sum(temp, axis=0)

            backward_matrix[:, time_stamp - 1] = backward_matrix[:, time_stamp - 1] / np.sum(
                backward_matrix[:, time_stamp - 1])
            assert backward_matrix.shape == (num_states, total_time_stamps)

        # print(backward_matrix)
        return backward_matrix

    def calculate_responsibility_matrix_1(self, forward_matrix, backward_matrix, f_sink):

        assert f_sink != 0.0, 'f_sink is equal to 0.0'
        assert forward_matrix.shape == backward_matrix.shape

        responsibility_matrix_1 = (forward_matrix * backward_matrix) / f_sink

        # normalize to make the column sum to 1
        for time_stamp in range(len(self.observations)):
            responsibility_matrix_1[:, time_stamp] = responsibility_matrix_1[:, time_stamp] \
                                                     / np.sum(responsibility_matrix_1[:, time_stamp])

        assert responsibility_matrix_1.shape == (self.num_states, len(self.observations))

        # print(responsibility_matrix_1[:, :5])

        return responsibility_matrix_1

    def calculate_responsibility_matrix_2(self, forward_matrix, backward_matrix, f_sink, gaussian_params,
                                          transition_probs):

        assert f_sink != 0.0, 'f_sink is equal to 0.0'
        assert forward_matrix.shape == backward_matrix.shape

        total_time_stamps = len(self.observations)

        responsibility_matrix_2 = np.zeros((self.num_states ** 2, total_time_stamps - 1))
        index = -1

        for state_1 in range(self.num_states):
            for state_2 in range(self.num_states):
                index += 1
                for time_stamp in range(total_time_stamps - 1):
                    emission_state_2 = self._get_emission_prob_EM(state_2, time_stamp, gaussian_params)
                    responsibility_matrix_2[index, time_stamp] = (forward_matrix[state_1, time_stamp] *
                                                                  transition_probs[state_1, state_2] *
                                                                  emission_state_2 *
                                                                  backward_matrix[state_2, time_stamp + 1]) / f_sink

                    # print(time_stamp, emission_state_2, state_1, state_2, responsibility_matrix_2[state_1, time_stamp])
        # normalize to make the column sum to 1

        for time_stamp in range(total_time_stamps - 1):
            responsibility_matrix_2[:, time_stamp] = responsibility_matrix_2[:, time_stamp] \
                                                     / np.sum(responsibility_matrix_2[:, time_stamp])

        return responsibility_matrix_2

    def baulm_welch_learn(self):
        epochs = 3
        # gaussian_params = np.random.randint(500, size=(self.gaussian_params.shape[0], self.gaussian_params.shape[1]))
        # transition_probs = np.random.rand(self.transition_probs.shape[0], self.transition_probs.shape[1])
        gaussian_params = self.gaussian_params
        transition_probs = self.transition_probs

        for _ in range(epochs):
            responsibility_matrix_1, responsibility_matrix_2 = self.e_step(transition_probs, gaussian_params)
            transition_probs, gaussian_params = self.m_step(responsibility_matrix_1, responsibility_matrix_2)

        print('The Parameters:', gaussian_params)
        print('Transitions:', transition_probs)

    def e_step(self, transition_probs, gaussian_parameters):
        forward_matrix, f_sink = self.calculate_forward_probs(transition_probs, gaussian_parameters)
        print('f_sink: ', f_sink)
        backward_matrix = self.calculate_backward_probs(transition_probs, gaussian_parameters)
        print('forward matrices:', forward_matrix[:, :50])
        print('backward matrices:', backward_matrix[:, :50])

        responsibility_matrix_1 = self.calculate_responsibility_matrix_1(forward_matrix, backward_matrix, f_sink)
        responsibility_matrix_2 = self.calculate_responsibility_matrix_2(forward_matrix, backward_matrix, f_sink,
                                                                         gaussian_parameters, transition_probs)

        return responsibility_matrix_1, responsibility_matrix_2

    def m_step(self, responsibility_matrix_1, responsibility_matrix_2):
        total_time_stamps = len(self.observations)

        assert responsibility_matrix_1.shape == (self.num_states, total_time_stamps)
        assert responsibility_matrix_2.shape == (self.num_states ** 2, total_time_stamps - 1)

        print('R1', responsibility_matrix_1)
        print('R2', responsibility_matrix_2)

        # we sum along the rows and get the estimated transition probabilities
        estimated_transition_probs = np.sum(responsibility_matrix_2, axis=1).reshape(self.num_states, self.num_states)

        print('Before', estimated_transition_probs)
        for state in range(self.num_states):
            estimated_transition_probs[state, :] = estimated_transition_probs[state, :] \
                                                     / np.sum(estimated_transition_probs[state, :])
        print(estimated_transition_probs)
        assert estimated_transition_probs.shape == self.transition_probs.shape

        mean = np.sum((responsibility_matrix_1 * self.observations), axis=1) / total_time_stamps
        mean = mean.reshape(mean.shape[0], 1)

        sigma = np.zeros((self.num_states, 1))

        for state in range(self.num_states):
            square_diff = (self.observations - mean[state]) ** 2
            sigma[state] = np.sum(responsibility_matrix_1[state, :] * square_diff) / total_time_stamps

        # print('sigma:', sigma.shape)

        estimated_gaussian_params = np.vstack((
            mean.T,
            sigma.T
        ))

        assert estimated_gaussian_params.shape == self.gaussian_params.shape
        return estimated_transition_probs, estimated_gaussian_params


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

    hmm.show_hmm_params()
    hmm.set_initial_probs()
    # hmm.generate_most_probable_states()
    hmm.baulm_welch_learn()
    # hmm.viterbi()
    # print(hmm.observations)
