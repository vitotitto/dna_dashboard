#import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt


#dataset creation part : 



def efficient_series_ofsequences(length_of_sequence, length_of_series, BASES):

    P = np.array([0.2, 0.3, 0.2, 0.3])
    # random DNA bases
    sequences_array = np.random.choice(BASES, (length_of_series, length_of_sequence), p=P)
    return sequences_array

def find_homopolymers(sequence):
    homopolymers = []
    for match in re.finditer(r"(A+|C+|T+|G+)", ''.join(sequence)):
        if len(match.group()) > 1:  # considering stretches 2 and longer
            homopolymers.append((match.start(), match.end()))
    return homopolymers


def introduce_errors_optimised(sequence, error_rate, BASES):
    BASES = np.array(['A', 'C', 'T', 'G'])
    error_types = ['insertion', 'mismatch', 'deletion']
    
    error_positions = {etype: [] for etype in error_types}
    
    num_errors = int(len(sequence) * error_rate)
    
    # bias errors towards the end
    probs = np.linspace(0.5, 1.5, len(sequence))
    
    # increasing error probability on homopolymers
    homopolymers = find_homopolymers(sequence)
    for start, end in homopolymers:
        probs[start:end] *= 2  # double the error probability for homopolymer regions
    
    probs /= probs.sum()
    
    positions = np.random.choice(len(sequence), size=num_errors, p=probs, replace=False)
    chosen_errors = np.random.choice(error_types, size=num_errors)
    
    # sortin positions and errors together based on positions for sequential processing
    sorted_indices = np.argsort(positions)
    positions = positions[sorted_indices]
    chosen_errors = chosen_errors[sorted_indices]
    
    shift = 0  # account for insertions and deletions
    
    for idx, pos in enumerate(positions):
        adjusted_pos = pos + shift
        error_type = chosen_errors[idx]
        
        if error_type == 'insertion':
            random_base = np.random.choice(BASES)
            sequence = np.insert(sequence, adjusted_pos, random_base)
            error_positions['insertion'].append(adjusted_pos)
            shift += 1  # account for the new insertion
            
        elif error_type == 'mismatch':
            original_base = sequence[adjusted_pos]
            replacement_bases = [base for base in BASES if base != original_base]
            sequence[adjusted_pos] = np.random.choice(replacement_bases)
            error_positions['mismatch'].append(adjusted_pos)
            
        elif error_type == 'deletion':
            sequence = np.delete(sequence, adjusted_pos)
            error_positions['deletion'].append(adjusted_pos)
            shift -= 1  # account for the deletion
    
    return sequence, error_positions

def run_code(error_rates, length_of_sequence, length_of_series, BASES):
    results = {}
    
    for rate in error_rates:
        corrupted_sequences_for_rate = []
        error_positions_for_rate = []
        sequence_set = efficient_series_ofsequences(length_of_sequence, length_of_series, BASES)
        
        for seq in sequence_set:
            corrupted_seq, error_positions = introduce_errors_optimised(seq, rate, BASES)
            corrupted_sequences_for_rate.append(corrupted_seq)
            error_positions_for_rate.append(error_positions)
        
        results[rate] = {
            'corrupted_sequences': corrupted_sequences_for_rate,
            'error_positions': error_positions_for_rate
        }
    
    return results



#plots : 
        
# plotting distributions (extra plot)

def plot_error_rates(error_rates, results):
    fig, axs = plt.subplots(ncols=len(error_rates), sharey=True, figsize=(10, 5))

    colors = {'insertion': 'blue', 'deletion': 'yellow', 'mismatch': 'green'}

    for i, rate in enumerate(error_rates):
        error_positions = results[rate]['error_positions']

        for error_type in ['insertion', 'deletion', 'mismatch']:
            positions = [pos for eps in error_positions for pos in eps[error_type]]
            axs[i].hist(positions, density=True, label=error_type, alpha=0.4, color=colors[error_type], edgecolor='black')

        axs[i].set_title(f'Error rate: {rate}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()      

        
# plotting errors positions : 




def plot_error_rates_pos(error_rates, results, error_types):
    for error_type in error_types:
        fig, axs = plt.subplots(ncols=len(error_rates), sharey=True, figsize=(7, 3))

        for i, rate in enumerate(error_rates):
            error_positions = results[rate]['error_positions']
            positions = [pos for eps in error_positions for pos in eps[error_type]]

            axs[i].hist(positions, density=True, label=error_type, alpha=0.4)
            axs[i].set_title(f'Error rate: {rate}')
            axs[i].legend()

        fig.suptitle(f'Distribution for {error_type.capitalize()} errors')
        plt.tight_layout()
        plt.show()


        
        
def plot_sequence_length_distribution_subplots(results):
    num_rates = len(results)
    
    fig, axes = plt.subplots(nrows=1, ncols=num_rates, figsize=(15, 5))
    
    if num_rates == 1:
        axes = [axes]
    
    for idx, rate in enumerate(results):
        lengths = [len(seq) for seq in results[rate]['corrupted_sequences']]
        axes[idx].hist(lengths, bins=range(0, max(lengths) + 2), alpha=0.7, color='skyblue', edgecolor='black')
        
        axes[idx].set_title(f"Error Rate: {rate*100}%")
        axes[idx].set_xlabel('Sequence Length')
        axes[idx].set_ylabel('Number of Sequences')
        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()




# average CG content : 

def compute_gc_content(sequence):
    """Computes the GC content of a sequence."""
    sequence = np.array(list(sequence))  # converting string sequence to numpy array for efficient computation
    num_g_c = np.sum(sequence == 'G') + np.sum(sequence == 'C')
    return (num_g_c / len(sequence)) * 100

def analyze_gc_content(results):
    """Analyze and store the average GC content and standard deviation for each error rate."""
    analysis = {}
    
    for rate in results:
        gc_contents = [compute_gc_content(seq) for seq in results[rate]['corrupted_sequences']]
        analysis[rate] = {
            'average_gc_content': np.mean(gc_contents),
            'std_dev_gc_content': np.std(gc_contents)
        }
    
    return analysis

def plot_gc_content_against_error_rate(analysis):
    """Plots the average GC content against the error rate."""
    error_rates = list(analysis.keys())
    avg_gc_contents = [analysis[rate]['average_gc_content'] for rate in error_rates]
    std_devs = [analysis[rate]['std_dev_gc_content'] for rate in error_rates]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(error_rates, avg_gc_contents, yerr=std_devs, fmt='-o', capsize=5, color='green')
    plt.title('Average GC Content vs. Error Rate')
    plt.xlabel('Error Rate')
    plt.ylabel('Average GC Content (%)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


#homopolymer ratios: 

def calc_homopolymer_ratios(sequences, error_rates):
    ratios = {rate:{'A':[], 'C':[], 'G':[], 'T':[]} for rate in error_rates}
    for rate in error_rates:
        for seq in sequences[rate]['corrupted_sequences']:
            hp = find_homopolymers(seq)
            total_len = len(seq)
            for start, end in hp:
                length = end - start
                base = seq[start]
                ratios[rate][base].append(length/total_len)
    return ratios
                
def plot_ratios(ratios, error_rates):
    plt.figure()
    for base in 'ACGT':
        means = [np.mean(ratios[rate][base]) for rate in error_rates]
        plt.plot(error_rates, means, label=base)
    plt.legend()
    plt.xlabel('Error Rate')
    plt.ylabel('Homopolymer Ratio')
    plt.title('Homopolymer Ratios by Error Rate')


#homopolymer position heatmap : 


def find_homopolymer_positions(sequence, BASES):
    """Finds positions of homopolymers for each nucleotide."""
    positions = {base: [] for base in BASES}
    for base in BASES:
        for match in re.finditer(f"{base}+", ''.join(sequence)):
            if len(match.group()) > 1:  # consider stretches 2 or longer
                positions[base].extend(range(match.start(), match.end()))
    return positions


def plot_homopolymer_positions_heatmap(results, BASES):
    error_rates = sorted(results.keys())
    sequence_length = len(results[error_rates[0]]['corrupted_sequences'][0])
    
    fig, axes = plt.subplots(nrows=len(error_rates), figsize=(10, 5 * len(error_rates)))
    
    for rate_idx, rate in enumerate(error_rates):
        homopolymer_frequencies = np.zeros((len(BASES), sequence_length))
        
        num_sequences = len(results[rate]['corrupted_sequences'])  
        
        for seq in results[rate]['corrupted_sequences']:
            positions = find_homopolymer_positions(seq, BASES)
            for base_idx, base in enumerate(BASES):
                for pos in positions[base]:
                    if pos < sequence_length:  #  position is within bounds
                        homopolymer_frequencies[base_idx, pos] += 1
        
        homopolymer_frequencies /= num_sequences
        
        # plotting
        ax = axes[rate_idx] if len(error_rates) > 1 else axes
        cax = ax.imshow(homopolymer_frequencies, aspect='auto', cmap='hot', origin='lower')
        fig.colorbar(cax, ax=ax, label='Frequency')
        ax.set_title(f'Homopolymer Positions for Error Rate: {rate*100}%')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Nucleotide')
        ax.set_yticks(np.arange(len(BASES)))
        ax.set_yticklabels(BASES)
    
    plt.tight_layout()
    plt.show()


#################################################

#running the code : 
def main():
    BASES = np.array(['A', 'C', 'T', 'G'])
    error_rates = [0.02, 0.05, 0.1]
    sequences = efficient_series_ofsequences(100, 200, BASES)
    results = run_code(error_rates, 100, 100, BASES)
    
    # error types
    error_types = ['insertion', 'deletion', 'mismatch']
    plot_error_rates(error_rates, results)
    plot_error_rates_pos(error_rates, results,error_types)
    plot_sequence_length_distribution_subplots(results)
    analysis = analyze_gc_content(results)
    plot_gc_content_against_error_rate(analysis)
    ratios = calc_homopolymer_ratios(results, error_rates) 
    plot_ratios(ratios, error_rates)
    plot_homopolymer_positions_heatmap(results, BASES)

if __name__ == "__main__":
    main()
#################################################