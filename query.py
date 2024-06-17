FILMS_TEMPLATE = """
?film <http://dbpedia.org/ontology/wikiPageWikiLink> <http://dbpedia.org/resource/Romantic_comedy> .
?film <http://dbpedia.org/ontology/wikiPageID> ?number .
?film <http://dbpedia.org/property/starring> ?starring .
?film <http://www.w3.org/2000/01/rdf-schema#comment> ?abstract .
?film <http://dbpedia.org/property/name> ?name .
OPTIONAL { ?film <http://dbpedia.org/ontology/cinematography> ?cinematography } .
OPTIONAL { ?film <http://dbpedia.org/ontology/director> ?director } .
OPTIONAL { ?film <http://dbpedia.org/ontology/gross> ?gross } .
OPTIONAL { ?film <http://dbpedia.org/ontology/producer> ?producer } .
OPTIONAL { ?film <http://dbpedia.org/property/language> ?language } .
"""

FILMS = f"""SELECT DISTINCT ?film, ?number, ?abstract, (GROUP_CONCAT(DISTINCT ?starring; SEPARATOR=";") AS ?starring), ?name, ?cinematography, ?director, ?gross, (GROUP_CONCAT(DISTINCT ?producer; SEPARATOR=";") AS ?producer), ?language
WHERE
     {{
        {FILMS_TEMPLATE}

        FILTER ( LANG ( ?abstract ) = 'en' )
      }}"""

CITIES_TEMPLATE = """
?city a <http://dbpedia.org/ontology/City> .
?city <http://dbpedia.org/ontology/populationTotal> ?population .
"""

CITIES = f"""
    SELECT ?city ?population
    WHERE {{
    
      {CITIES_TEMPLATE}
    
    }}
"""


GROUPS_TEMPLATE = """
?music_band <http://dbpedia.org/property/name> ?name.
?music_band <http://dbpedia.org/property/origin> ?origin.
?music_band <http://dbpedia.org/ontology/activeYearsStartYear> ?creation_date.
?music_band <http://dbpedia.org/ontology/genre> ?genre.
?music_band <http://dbpedia.org/ontology/bandMember> ?band_member.
?song <http://dbpedia.org/property/artist> ?music_band.
"""

GROUPS = f"""
SELECT ?music_band ?name ?origin ?creation_date (GROUP_CONCAT(DISTINCT ?genre; SEPARATOR=";") AS ?genre) (GROUP_CONCAT(DISTINCT ?band_member; SEPARATOR=";") AS ?band_member) (GROUP_CONCAT(DISTINCT ?song; SEPARATOR=";") AS ?song)
WHERE {{
{GROUPS_TEMPLATE}
}}
"""

UNIVERSITIES_TEMPLATE = """
?uni <http://dbpedia.org/ontology/type> <http://dbpedia.org/resource/Public_university>.
?uni <http://dbpedia.org/ontology/city> ?city.
?city <http://dbpedia.org/property/name> ?name_of_city.
?city <http://dbpedia.org/ontology/populationTotal> ?population_of_city.
OPTIONAL { ?uni <http://dbpedia.org/ontology/numberOfDoctoralStudents> ?doctoral_students } .
OPTIONAL { ?uni <http://dbpedia.org/ontology/numberOfPostgraduateStudents> ?postgraduate_students } .
OPTIONAL { ?uni <http://dbpedia.org/ontology/numberOfUndergraduateStudents> ?undergraduate_students } .
"""

UNIVERSITIES = f"""
SELECT ?uni ?city ?name_of_city ?population_of_city ?doctoral_students ?postgraduate_students ?undergraduate_students
WHERE {{
{UNIVERSITIES_TEMPLATE}
}}
"""

BOOKS_TEMPLATE = """
#?book <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>	<http://dbpedia.org/ontology/Book> .
?book <http://dbpedia.org/ontology/author>	?author .
?book <http://dbpedia.org/ontology/abstract> ?abstract .
?book <http://dbpedia.org/property/followedBy> ?next_book .
?book <http://dbpedia.org/ontology/literaryGenre> ?genre .
?book <http://dbpedia.org/property/setIn> ?setin_location .
"""
"""
r#, ?abstract, ?next_book, ?genre, ?setin_location

?book <https://dbpedia.org/ontology/abstract> ?abstract .
?book <https://dbpedia.org/property/followedBy> ?next_book .
?book <https://dbpedia.org/ontology/literaryGenre> ?genre .
?book <https://dbpedia.org/property/setIn> ?setin_location .



        #FILTER ( LANG ( ?abstract ) = 'en' )
"""
BOOKS = f"""SELECT DISTINCT ?book, ?author, ?abstract, (GROUP_CONCAT(DISTINCT ?next_book; SEPARATOR=";") AS ?next_book), (GROUP_CONCAT(DISTINCT ?genre; SEPARATOR=";") AS ?genre), (GROUP_CONCAT(DISTINCT ?setin_location; SEPARATOR=";") AS ?setin_location)
WHERE
     {{
        {BOOKS_TEMPLATE}
        FILTER ( LANG ( ?abstract ) = 'en' )
      }}"""

ALL = {
    "films": {
        "query": FILMS,
        "template": FILMS_TEMPLATE,
    },
    "cities": {
        "query": CITIES,
        "template": CITIES_TEMPLATE,
    },
    "universities": {
        "query": UNIVERSITIES,
        "template": UNIVERSITIES_TEMPLATE,
    },
    "groups": {
        "query": GROUPS,
        "template": GROUPS_TEMPLATE,
    },
    "books": {
        "query": BOOKS,
        "template": BOOKS_TEMPLATE,
    }
}