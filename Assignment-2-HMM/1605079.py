import numpy as np
import os
from statistics import NormalDist
import argparse

np.random.seed(79)


class HMM:
    def __init__(self, transition_probs, gaussian_params, num_states, observations, output_path):
        self.transition_probs = transition_probs
        self.gaussian_params = gaussian_params
        self.num_states = num_states
        self.observations = observations
        self.output_path = output_path
        self.initial_state = []

    def show_hmm_params(self):
        print(f'Number of States {self.num_states}')
        print(f'Transition Probabilities:\n {self.transition_probs}')
        print(f'Transition Probabilities Shape: {self.transition_probs.shape}')
        print(f'Gaussian Params\n {self.gaussian_params}')
        print(f'Gaussian Params Shape:\n {self.gaussian_params.shape}')
        print(f'Observation Params Shape:\n {self.observations.shape}')

    def set_initial_probs(self, transition_probs):
        # modifying the transition probability matrices by subtracting diagonal elements
        transition_probs = np.copy(transition_probs).T
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

        # setting the initial state of the probability matrix and the max state tracker matrix
        for state_no in range(self.num_states):
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

                # checking to see if any value is zero before passing to log
                assert np.all(self.transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state)

                # calculate temp value by slicing
                temp = probability_matrix[:, time_stamp - 1].reshape(num_states, 1) \
                       + np.log((self.transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state))

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

        print('Viterbi Algorithm Ran!')
        return hidden_states, probability_matrix, max_state_index_tracker

    def generate_most_probable_states(self, with_learning=False):
        output_file_name = 'output_learning.txt' if with_learning else 'output_wo_learning.txt'
        hidden_states, probability_matrix, state_index = self.run_viterbi()
        print('Generating Most Probable Sequence and writing it to ./Output/')

        # self.write_to_file(probability_matrix, 'viterbi_probability.txt')
        # self.write_to_file(state_index, 'state.txt')

        most_probable_states = ["\"El Nino\"\n" if state == 0 else "\"La Nina\"\n" for state in hidden_states]

        output = open(f'{self.output_path}/{output_file_name}', 'w+')

        output.writelines(most_probable_states)
        print()

    def _get_emission_prob(self, state_no, time_stamp):
        prob_dist_params = self.gaussian_params[:, state_no]
        return NormalDist(mu=prob_dist_params[0], sigma=np.sqrt(prob_dist_params[1])). \
            pdf(self.observations[time_stamp])

    def _get_emission_prob_EM(self, state_no, time_stamp, gaussian_params):
        # print('In here', gaussian_params, time_stamp, state_no)
        prob_dist_params = gaussian_params[:, state_no]
        return NormalDist(mu=prob_dist_params[0], sigma=np.sqrt(prob_dist_params[1])). \
            pdf(self.observations[time_stamp])

    def calculate_forward_probs(self, transition_probs, gaussian_parameters):
        total_time_stamps = len(self.observations)
        """
        we set up one matrix to keep track of the outcome likelihood
        """
        forward_matrix = np.zeros((self.num_states, total_time_stamps), dtype=np.float64)

        self.set_initial_probs(transition_probs)
        # print(self.initial_state)
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

                # calculate temp value by slicing
                temp = forward_matrix[:, time_stamp - 1].reshape(num_states, 1) * \
                       transition_probs[:, state_no].reshape(num_states, 1) * emission_current_state

                assert temp.shape == (num_states, 1)

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
            for current_state in range(self.num_states):
                temp_sum = 0
                for incoming_state in range(self.num_states):
                    # first calculate the emission probability from gaussian pdf
                    emission_incoming_state = self._get_emission_prob_EM(state_no=incoming_state, time_stamp=time_stamp,
                                                                         gaussian_params=gaussian_parameters)

                    """
                    for state a:
                        for updating a2 we calculate the following
                        temp = b[a3, b3] * T[a3->a2, b3->a2] * e_a3 (element wise product)
                        temp (2,1)
                        b[a2] = np.sum(temp [something, something], axis=column wise)
                    """
                    temp_sum += backward_matrix[incoming_state, time_stamp] * \
                                transition_probs[current_state, incoming_state] * emission_incoming_state

                    # summing along the axes of the temp vector
                backward_matrix[current_state, time_stamp - 1] = temp_sum

            backward_matrix[:, time_stamp - 1] = backward_matrix[:, time_stamp - 1] / np.sum(
                backward_matrix[:, time_stamp - 1])
            assert backward_matrix.shape == (num_states, total_time_stamps)

        # print(backward_matrix)
        return backward_matrix

    def calculate_responsibility_matrix_1(self, forward_matrix, backward_matrix, f_sink):

        assert f_sink != 0.0, 'f_sink is equal to 0.0'
        assert forward_matrix.shape == backward_matrix.shape

        responsibility_matrix_1 = (forward_matrix * backward_matrix) / f_sink
        responsibility_matrix_1 = self._normalize_along_timestamp(responsibility_matrix_1, len(self.observations))

        assert responsibility_matrix_1.shape == (self.num_states, len(self.observations))
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
                    emission_state_2 = self._get_emission_prob_EM(state_2, time_stamp + 1, gaussian_params)
                    responsibility_matrix_2[index, time_stamp] = (forward_matrix[state_1, time_stamp] *
                                                                  transition_probs[state_1, state_2] *
                                                                  emission_state_2 *
                                                                  backward_matrix[state_2, time_stamp + 1]) / f_sink

                    # print(time_stamp, emission_state_2, state_1, state_2, responsibility_matrix_2[state_1, time_stamp])
        # normalize to make the column sum to 1
        responsibility_matrix_2 = self._normalize_along_timestamp(responsibility_matrix_2, total_time_stamps - 1)

        return responsibility_matrix_2

    def baum_welch_learn(self, useRandom=False, epochs=10):

        if useRandom:
            # print('In Random')
            learned_gaussian_params = np.random.randint(low=80, high=180, size=(
            self.gaussian_params.shape[0], self.gaussian_params.shape[1]))
            learned_transition_probs = np.random.uniform(low=0, high=1, size=(
            self.transition_probs.shape[0], self.transition_probs.shape[1]))
            learned_transition_probs = learned_transition_probs / np.sum(learned_transition_probs, axis=1)[:,
                                                                  np.newaxis]
            # print(learned_transition_probs)
            # print(learned_gaussian_params)
        else:
            learned_gaussian_params = self.gaussian_params
            learned_transition_probs = self.transition_probs
        # print('initial', learned_transition_probs)
        # print(learned_gaussian_params)

        # responsibility_matrix_1, responsibility_matrix_2 = self.e_step(learned_transition_probs, learned_gaussian_params)
        # learned_transition_probs, learned_gaussian_params = self.m_step(responsibility_matrix_1,
        #                                                                 responsibility_matrix_2)

        prev_gaussian_params = learned_gaussian_params
        prev_transitions = learned_transition_probs

        for iteration in range(epochs):

            if iteration % 10 == 0 and iteration != 0:
                print(f'Ran {iteration} iteration')
            responsibility_matrix_1, responsibility_matrix_2 = self.e_step(learned_transition_probs,
                                                                           learned_gaussian_params)

            learned_transition_probs, learned_gaussian_params = self.m_step(responsibility_matrix_1,
                                                                            responsibility_matrix_2)

            convergence_criteria = np.sum(np.abs(prev_transitions - learned_transition_probs)) \
                                   + np.sum(np.abs(prev_gaussian_params - learned_gaussian_params))

            # print('convergence criteria', convergence_criteria)
            if convergence_criteria < 0.00001:
                print(f'Parameter Estimation Converged at {iteration} iteration')
                break

            # if np.abs(prev_gaussian_params[0, 0] - learned_gaussian_params[0, 0]) < 0.000001:
            #     print(f'Breaking After {iteration} iteration')
            #     break

            prev_gaussian_params = learned_gaussian_params
            prev_transitions = learned_transition_probs

        # print(iteration)
        # print('The Parameters:', learned_gaussian_params)
        # print('Transitions:', learned_transition_probs)

        self.generate_parameter_output_file(learned_transition_probs, learned_gaussian_params)

        print('Baum-Welch Algorithm Ran. Writing Learned Parameters to ./Output/')

        return learned_transition_probs, learned_gaussian_params

    def e_step(self, transition_probs, gaussian_parameters):

        assert transition_probs.shape == (self.num_states, self.num_states)
        assert gaussian_parameters.shape == (2, self.num_states)

        forward_matrix, f_sink = self.calculate_forward_probs(transition_probs, gaussian_parameters)
        # print('f_sink: ', f_sink)
        backward_matrix = self.calculate_backward_probs(transition_probs, gaussian_parameters)
        # print('forward matrices:', forward_matrix[:, :50])
        # print('backward matrices:', backward_matrix[:, :50])

        # self.write_to_file(forward_matrix, 'forward.txt')
        # self.write_to_file(backward_matrix, 'backward.txt')

        responsibility_matrix_1 = self.calculate_responsibility_matrix_1(forward_matrix, backward_matrix, f_sink)
        responsibility_matrix_2 = self.calculate_responsibility_matrix_2(forward_matrix, backward_matrix, f_sink,
                                                                         gaussian_parameters, transition_probs)

        # self.write_to_file(responsibility_matrix_1, 'pi_star.txt')
        # self.write_to_file(responsibility_matrix_2, 'pi_star_star.txt')

        return responsibility_matrix_1, responsibility_matrix_2

    def m_step(self, responsibility_matrix_1, responsibility_matrix_2):
        total_time_stamps = len(self.observations)

        assert responsibility_matrix_1.shape == (self.num_states, total_time_stamps)
        assert responsibility_matrix_2.shape == (self.num_states ** 2, total_time_stamps - 1)

        # we sum along the rows and get the estimated transition probabilities
        estimated_transition_probs = np.sum(responsibility_matrix_2, axis=1).reshape(self.num_states, self.num_states)
        estimated_transition_probs = self._normalize_along_state(estimated_transition_probs)

        assert estimated_transition_probs.shape == self.transition_probs.shape

        mean = np.sum((responsibility_matrix_1 * self.observations), axis=1) / np.sum(responsibility_matrix_1, axis=1)
        mean = mean.reshape(mean.shape[0], 1)

        sigma = np.zeros((self.num_states, 1))

        for state in range(self.num_states):
            square_diff = (self.observations - mean[state]) ** 2
            sigma[state] = np.sum(responsibility_matrix_1[state, :] * square_diff) / np.sum(
                responsibility_matrix_1[state, :])

        # print('sigma:', sigma.shape)

        estimated_gaussian_params = np.vstack((
            mean.T,
            sigma.T
        ))

        assert estimated_gaussian_params.shape == self.gaussian_params.shape
        return estimated_transition_probs, estimated_gaussian_params

    def _normalize_along_state(self, matrix):
        for state in range(self.num_states):
            matrix[state, :] = matrix[state, :] / np.sum(matrix[state, :])

        return matrix

    @staticmethod
    def _normalize_along_timestamp(matrix, total_time_stamps):
        # normalize to make the column sum to 1
        for time_stamp in range(total_time_stamps):
            matrix[:, time_stamp] = matrix[:, time_stamp] / np.sum(matrix[:, time_stamp])

        return matrix

    def write_to_file(self, matrix, f_name):
        out = open(
            f'{self.output_path}/{f_name}',
            'w+')

        for col in range(matrix.shape[1]):
            data = ' '.join([str(val) for val in matrix[:, col]])
            out.write(data + '\n')

    def set_hmm_params(self, learned_transitions, learned_gaussian_params):
        self.transition_probs = learned_transitions
        self.gaussian_params = learned_gaussian_params

    def generate_parameter_output_file(self, learned_transition_probs, learned_gaussian_params):
        parameter_file = open(
            f'{self.output_path}/Parameters_learned.txt',
            'w+')

        parameter_file.write(str(self.num_states) + '\n')

        for state in range(self.num_states):
            parameter_file.write('    '.join([str(t) for t in learned_transition_probs[state, :]]) + '\n')

        for index in range(2):
            parameter_file.write('    '.join([str(t) for t in learned_gaussian_params[index, :]]) + '\n')

        parameter_file.write('    '.join([str(p) for p in self.initial_state]) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='path to input file')

    # Add the arguments
    parser.add_argument('--i',
                        metavar='i',
                        type=str,
                        help='the path to the input file')

    parser.add_argument('--p',
                        metavar='p',
                        type=str,
                        help='the path to the parameters file')

    parser.add_argument('--alg',
                        metavar='p',
                        type=str,
                        help='the algorithm to run')

    # Execute the parse_args() method
    args = parser.parse_args()

    input_file_path = args.i
    parameter_file_path = args.p
    algorithm = args.alg

    observations = open(input_file_path, 'r').readlines()
    parameters = open(parameter_file_path, 'r').readlines()

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

    output_directory_path = './Output'

    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    hmm = HMM(
        transition_probs=np.array(transition_probs),
        gaussian_params=np.array(gaussian_params),
        num_states=num_states,
        observations=np.array(observations),
        output_path=output_directory_path
    )

    # hmm.show_hmm_params()

    if algorithm == 'viterbi':
        hmm.set_initial_probs(hmm.transition_probs)
        hmm.generate_most_probable_states()
    elif algorithm == 'baum-welch':
        # viterbi learning
        estimated_transition_probs, estimated_gaussian_params = hmm.baum_welch_learn(useRandom=True, epochs=10)
        hmm.set_hmm_params(estimated_transition_probs, estimated_gaussian_params)
        hmm.generate_most_probable_states(with_learning=True)
    else:
        hmm.set_initial_probs(hmm.transition_probs)
        hmm.generate_most_probable_states()

        # viterbi learning
        estimated_transition_probs, estimated_gaussian_params = hmm.baum_welch_learn(useRandom=False, epochs=10)
        hmm.set_hmm_params(estimated_transition_probs, estimated_gaussian_params)
        hmm.generate_most_probable_states(with_learning=True)
