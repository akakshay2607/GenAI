from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template='Generate notes from the given  text.\n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate quiz from the given  text.\n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quizes into single document.\n notes -> {notes} quize -> {quize}',
    input_variables=['text']
)

parser = StrOutputParser()

parallel_chains = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quize' : prompt2 | model | parser
})

second_chain = prompt3 | model | parser

merged_chain = parallel_chains | second_chain

result = merged_chain.invoke("""
                    The oldest artifacts considered pornographic were discovered in Germany in 2008 and are dated to be at least 35,000 years old.[b] Human enchantment with sexual imagery representations has been a constant throughout history. However, the reception of such imagery varied according to the historical, cultural, and national contexts. The Indian Sanskrit text Kama Sutra (3rd century CE) contained prose, poetry, and illustrations regarding sexual behavior, and the book was celebrated; while the British English text Fanny Hill (1748), considered "the first original English prose pornography," has been one of the most prosecuted and banned books. In the late 19th century, a film by Thomas Edison that depicted a kiss was denounced as obscene in the United States, whereas Eugène Pirou's 1896 film Bedtime for the Bride was received very favorably in France. Starting from the mid-twentieth century on, societal attitudes towards sexuality became lenient in the Western world where legal definitions of obscenity were made limited. In 1969, Blue Movie by Andy Warhol became the first film to depict unsimulated sex that received a wide theatrical release in the United States. This was followed by the "Golden Age of Porn" (1969–1984). The introduction of home video and the World Wide Web in the late 20th century led to global growth in the pornography business. Beginning in the 21st century, greater access to the Internet and affordable smartphones made pornography more mainstream.
Pornography has been vouched to provision a safe outlet for sexual desires that may not be satisfied within relationships and be a facilitator of sexual fulfillment in people who do not have a partner. Pornography consumption is found to induce psychological moods and emotions similar to those evoked during sexual intercourse and casual sex. Pornography usage is considered a widespread recreational activity in-line with other digitally mediated activities such as use of social media or video games.[c] People who regard porn as sex education material were identified as more likely not to use condoms in their own sex life, thereby assuming a higher risk of contracting sexually transmitted infections (STIs); performers working for pornographic studios undergo regular testing for STIs unlike much of the general public. Comparative studies indicate higher tolerance and consumption of pornography among adults tends to be associated with their greater support for gender equality. Among feminist groups, some seek to abolish pornography believing it to be harmful, while others oppose censorship efforts insisting it is benign. A longitudinal study ascertained pornography use is not a predictive factor in intimate partner violence.[d] Porn Studies, started in 2014, is the first international peer-reviewed, academic journal dedicated to critical study of pornographic "products and services".
Pornography is a major influencer of people's perception of sex in the digital age; numerous pornographic websites rank among the top 50 most visited websites worldwide. Called an "erotic engine", pornography has been noted for its key role in the development of various communication and media processing technologies. For being an early adopter of innovations and a provider of financial capital, the pornography industry has been cited to be a contributing factor in the adoption and popularization of media related technologies. The exact economic size of the porn industry in the early twenty-first century is unknown. In 2023, estimates of the total market value stood at over US$172 billion. The legality of pornography varies across countries. People hold diverse views on the availability of pornography. From the mid-2010s, unscrupulous pornography such as deepfake pornography and revenge porn have become issues of concern.
""")


merged_chain.get_graph().print_ascii()