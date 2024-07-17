import mido
import time

# Define melody and company parts
melody = [('C', 5, 1/3), ('G', 5, 1/3), ('E', 5, 1/3),
          ('C', 5, 1/3), ('G', 5, 1/3), ('E', 5, 1/3),
          ('C', 5, 1/3), ('F', 5, 1/3), ('E', 5, 1/3),
          ('D', 5, 1),
          ('D', 5, 1/3), ('F', 5, 1/3), ('E', 5, 1/3),
          ('D', 5, 1/3), ('F', 5, 1/3), ('E', 5, 1/3),
          ('D', 5, 1/3), ('E', 5, 1/3), ('D', 5, 1/3),
          ('C', 5, 1),
          ('C', 5, 1/3), ('G', 5, 1/3), ('E', 5, 1/3),
          ('C', 5, 1/3), ('G', 5, 1/3), ('E', 5, 1/3),
          ('C', 5, 1/3), ('D', 5, 1/3), ('E', 5, 1/3),
          ('F', 5, 1),
          ('F', 5, 1/3), ('G', 5, 1/3), ('F', 5, 1/3),
          ('E', 5, 1/3), ('F', 5, 1/3), ('E', 5, 1/3),
          ('D', 5, 1/3), ('E', 5, 1/3), ('D', 5, 1/3),
          ('C', 5, 1),
          ]
company = [('C', 4, 2), ('G', 3, 2), ('C', 5, 2)]

# Define tempo and output file name
tempo = 120
output_file = '/Users/ericliao/Desktop/melody.mid'

# Define MIDI message parameters
channel = 0
velocity = 64
program = 0  # acoustic grand piano

# Initialize MIDI track and time counter
track = mido.MidiTrack()
time = 0

# Set program (instrument)
program_message = mido.Message('program_change', program=program, time=0, channel=channel)
track.append(program_message)

# Convert note name to MIDI note number
def note_to_midi(note):
    octave = int(note[-1])
    note_name = note[:-1]
    midi_note = 12 + 12 * octave

    if note_name == 'C':
        midi_note += 0
    elif note_name == 'C#':
        midi_note += 1
    elif note_name == 'D':
        midi_note += 2
    elif note_name == 'D#':
        midi_note += 3
    elif note_name == 'E':
        midi_note += 4
    elif note_name == 'F':
        midi_note += 5
    elif note_name == 'F#':
        midi_note += 6
    elif note_name == 'G':
        midi_note += 7
    elif note_name == 'G#':
        midi_note += 8
    elif note_name == 'A':
        midi_note += 9
    elif note_name == 'A#':
        midi_note += 10
    elif note_name == 'B':
        midi_note += 11

    return midi_note

# Loop through melody and company parts
for note, octave, duration in melody + company:
    note_value = note_to_midi(note + str(octave))
    note_on = mido.Message('note_on', note=note_value, velocity=velocity, time=int(time))
    note_off = mido.Message('note_off', note=note_value, velocity=velocity, time=int(duration * 60 / tempo * 1000))
    track.append(note_on)
    track.append(note_off)
    time = 0

# Create MIDI file and add track
midi_file = mido.MidiFile()
midi_file.tracks.append(track)

# Save MIDI file
midi_file.save(output_file)
