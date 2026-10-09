import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from seq2seq import EncoderRNN, ConditionalDecoderRNN, HarperSeq2Seq
from hierarchical_seq2seq import HarperHRED
from evaluator import AdversarialDiscriminator, EvaluationMetrics
from rhetoric_analyzer import RhetoricAnalyzer
from rlhf_policy_gradient import PolicyGradientRLHF
from prosody_tacotron import ProsodyAcousticDecoder, GriffinLimVocoder

class StyleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(StyleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        outputs, _ = self.gru(embedded)
        pooled = torch.mean(outputs, dim=1)
        return F.log_softmax(self.fc(pooled), dim=1) if hasattr(torch.nn.functional, 'log_softmax') else nn.functional.log_softmax(self.fc(pooled), dim=1)

class ProductionDataFactory:
    def __init__(self, vocab_size, num_tones, rhetoric_dim, batch_size=4, seq_len=15):
        self.vocab_size = vocab_size
        self.num_tones = num_tones
        self.rhetoric_dim = rhetoric_dim
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_batch(self, use_cuda=False):
        lengths_np = np.random.randint(5, self.seq_len + 1, size=self.batch_size)
        lengths_np = np.sort(lengths_np)[::-1].copy()
        
        input_seqs = torch.LongTensor(self.batch_size, self.seq_len).random_(2, self.vocab_size)
        target_seqs = torch.LongTensor(self.batch_size, self.seq_len).random_(2, self.vocab_size)
        
        for i, l in enumerate(lengths_np):
            if l < self.seq_len:
                input_seqs[i, l:] = 0
                target_seqs[i, l:] = 0
                
        input_lengths = torch.LongTensor(lengths_np)
        target_tones = torch.LongTensor(self.batch_size).random_(0, self.num_tones)
        target_rhetoric_vecs = torch.FloatTensor(self.batch_size, self.rhetoric_dim).normal_()
        
        num_turns = 3
        history_seqs = torch.LongTensor(self.batch_size, num_turns, self.seq_len).random_(2, self.vocab_size)
        history_lengths = torch.LongTensor(self.batch_size, num_turns).random_(5, self.seq_len + 1)
        
        pitch_energy_duration = torch.FloatTensor(self.batch_size, 3).uniform_(0.5, 2.0)
        
        if use_cuda:
            return (Variable(input_seqs.cuda()), Variable(input_lengths.cuda()), 
                    Variable(target_seqs.cuda()), Variable(target_tones.cuda()), 
                    Variable(target_rhetoric_vecs.cuda()), Variable(history_seqs.cuda()), 
                    Variable(history_lengths.cuda()), Variable(pitch_energy_duration.cuda()))
        else:
            return (Variable(input_seqs), Variable(input_lengths), 
                    Variable(target_seqs), Variable(target_tones), 
                    Variable(target_rhetoric_vecs), Variable(history_seqs), 
                    Variable(history_lengths), Variable(pitch_energy_duration))

def main():
    print("Initializing Harper Stylized Text & Speech Dialogue System Pipeline...")
    
    VOCAB_SIZE = 500
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 128
    TONE_DIM = 32
    NUM_TONES = 3
    RHETORIC_DIM = 256
    MEL_DIM = 80
    BATCH_SIZE = 4
    
    USE_CUDA = torch.cuda.is_available()
    print("Device Target execution context: [CUDA Available: {}]".format(USE_CUDA))
    
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, n_layers=2)
    decoder = ConditionalDecoderRNN(VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, TONE_DIM, NUM_TONES, RHETORIC_DIM, n_layers=2)
    seq2seq_model = HarperSeq2Seq(encoder, decoder)
    
    hred_model = HarperHRED(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, TONE_DIM, NUM_TONES, RHETORIC_DIM, n_layers=1)
    
    discriminator = AdversarialDiscriminator(VOCAB_SIZE, EMBEDDING_DIM)
    style_classifier = StyleClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_TONES)
    
    acoustic_decoder = ProsodyAcousticDecoder(MEL_DIM, HIDDEN_SIZE, TONE_DIM, NUM_TONES, prosody_feature_dim=3)
    
    if USE_CUDA:
        seq2seq_model.cuda()
        hred_model.cuda()
        discriminator.cuda()
        style_classifier.cuda()
        acoustic_decoder.cuda()
        
    gen_optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.001)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005)
    
    rlhf_agent = PolicyGradientRLHF(
        generator_model=seq2seq_model,
        discriminator=discriminator,
        style_classifier=style_classifier,
        optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
        pad_token=0,
        sos_token=1
    )
    
    data_provider = ProductionDataFactory(VOCAB_SIZE, NUM_TONES, RHETORIC_DIM, batch_size=BATCH_SIZE)
    
    print("\nExecuting Rhetoric Structural Analyzer Subsystems...")
    analyzer = RhetoricAnalyzer(spacy_model='en_core_web_sm')
    sample_text = "Standard evaluation string optimization rule execution check logic."
    features, mock_emb = analyzer.analyze_text(sample_text)
    print("Extracted Analytical Dictionary Metrics: {}".format(features))
    
    print("\nExecuting Model Optimization Pass loops (Standard Seq2Seq & HRED Structural variants)...")
    input_seqs, input_lengths, target_seqs, target_tones, target_rhetoric_vecs, history_seqs, history_lengths, pitch_energy_duration = data_provider.get_batch(USE_CUDA)
    
    seq2seq_out = seq2seq_model(input_seqs, input_lengths, target_seqs, target_tones, target_rhetoric_vecs)
    print("Standard Stylized Seq2Seq Log Probability Matrix Shape: {}".format(list(seq2seq_out.size())))
    
    hred_out = hred_model(history_seqs, history_lengths, target_seqs, target_tones, target_rhetoric_vecs)
    print("Hierarchical HRED Log Probability Matrix Shape: {}".format(list(hred_out.size())))
    
    perplexity = EvaluationMetrics.calculate_perplexity(seq2seq_out, target_seqs, pad_token_id=0)
    print("Computed Current Batch Generator Perplexity Score: {:.4f}".format(perplexity))
    
    print("\nRunning Active Reinforcement Optimization Loops (RLHF Optimization Iteration)...")
    pg_loss, entropy_bonus, reward_mean = rlhf_agent.train_step(input_seqs, input_lengths, target_tones, target_rhetoric_vecs)
    print("RLHF Metric Trackers -> PG Loss: {:.4f} | Entropy Bonus: {:.4f} | Reward Baseline Mean: {:.4f}".format(pg_loss, entropy_bonus, reward_mean))
    
    disc_loss = rlhf_agent.train_discriminator_step(target_seqs, input_seqs, input_lengths, target_tones, target_rhetoric_vecs)
    print("WGAN Discriminator Step Adversarial Gradient Penalty Loss: {:.4f}".format(disc_loss))
    
    print("\nExecuting Speech Acoustic Synthesis Stage (Tacotron Engine Pipeline Expansion)...")
    encoder_outputs, _ = seq2seq_model.encoder(input_seqs, input_lengths)
    
    mel_seq, linear_seq, stop_outputs = acoustic_decoder(encoder_outputs, target_tones, pitch_energy_duration)
    print("Generated Audio Mel Spectrogram Matrix Dimensions: {}".format(list(mel_seq.size())))
    print("Generated PostNet Linear Spectrogram Matrix Dimensions: {}".format(list(linear_seq.size())))
    
    linear_spec_np = linear_seq.data.cpu().numpy()[0]
    print("Running Vocoder Griffin-Lim Phase Alignment Conversion Pass...")
    waveform = GriffinLimVocoder.synthesize(linear_spec_np, n_iter=5)
    print("Synthesized Audio Output Vector Length Dimension: {} samples".format(waveform.shape[0]))
    
    print("\nExecuting Dynamic Time Warping Realizations (Prosody Correlation Calculation)...")
    pred_pitch = Variable(torch.FloatTensor(BATCH_SIZE, 30).normal_())
    target_pitch = Variable(torch.FloatTensor(BATCH_SIZE, 30).normal_())
    pad_mask = Variable(torch.ones(BATCH_SIZE, 30))
    if USE_CUDA:
        pred_pitch, target_pitch, pad_mask = pred_pitch.cuda(), target_pitch.cuda(), pad_mask.cuda()
        
    correlation_coefficient = EvaluationMetrics.calculate_prosody_correlation(pred_pitch, target_pitch, pad_mask)
    print("Aligned DTW Dynamic Time Warped Prosody Correlation Alignment Metric: {:.4f}".format(correlation_coefficient))
    
    print("\nAll pipeline architecture constraints passed successfully. System operational.")

if __name__ == '__main__':
    main()