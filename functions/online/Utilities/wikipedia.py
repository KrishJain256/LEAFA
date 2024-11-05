import wikipedia as wiki

def search_on_wikipedia(query,length):
    results = wiki.summary(query, sentences=length)
    return results

print(search_on_wikipedia("Chess",length=4))