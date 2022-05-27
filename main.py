from caption_generation import generate_caption, generate_sentence

n = int(input('How many times would you like to see ? -->'))

for i in range(n):
    arabe_caption = generate_caption(
        'https://www.preventivevet.com/hubfs/Three%20dogs%20playing%20in%20the%20yard%20600%20canva.jpg')
    generate_sentence(meter='الكامل', rhyme='ر', start_with=arabe_caption)
