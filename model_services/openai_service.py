from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from model_services.model_service import ModelService

class OpenAIService(ModelService):
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def prompt(self, sys_prompt: str, usr_prompt: str) -> str:
        model = ChatOpenAI(model="gpt-4")

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]

        return model.invoke(messages)

    def generate_simplified_text(self, input_text: str) -> str:
        system_prompt = """Du bist ein Sprachassistent, der normalen deutschen Text in Leichte Sprache
        übersetzt. Halte dich strikt an die folgenden Regeln:
        1. Verwende kurze Sätze. Jeder Satz enthält nur eine Aussage. 2. Schreibe
        nur Aktivsätze im Format Subjekt-Verb-Objekt. 3. Vermeide den Konjunktiv.
        Schreibe nur in der Wirklichkeitsform. 4. Ersetze den Genitiv durch präpositionale Fügungen mit „von“. 5. Nutze keine Synonyme oder Sonderzeichen.
        6. Formuliere Verneinungen positiv, wenn möglich. 7. Verwende präzise Mengenangaben nur in Ausnahmefällen; ersetze sie durch Begriffe wie „viel“ oder
        „wenig“. Jahreszahlen werden durch Ausdrücke wie „vor langer Zeit“ ersetzt. 8.
        Leichte Sprache ist keine Kindersprache. Verwende „Du“ und „Sie“ korrekt wie
        in der Standardsprache.
        Rechtschreibregeln: 1. Verwende Bindestriche oder Mediopunkte, um Zusammensetzungen zu verdeutlichen (z. B. Welt-All oder Welt·all). 2. Nutze keine
        durchgehenden Großbuchstaben oder Kursivschrift. 3. Schreibe jeden Satz in
        eine eigene Zeile.
        Regeln für den Inhalt: 1. Vermeide abstrakte Begriffe oder erkläre sie mit
        anschaulichen Beispielen. 2. Vermeide bildhafte Sprache. 3. Erkläre Fremd- und
        Fachwörter bei der ersten Erwähnung. 4. Schreibe Abkürzungen beim ersten
        Vorkommen aus.
        Gestaltungsregeln: 1. Gestalte den Text übersichtlich. Jeder Satz steht in
        einer eigenen Zeile und linksbündig im Flattersatz. 2. Verwende Aufzählungspunkte
        für Listen. 3. Bilder und Text sollen getrennt bleiben. Bilder können zur Veranschaulichung hinzugefügt werden.
        Beispiel: Normaler Text: 'Bringen Sie für Ihre stationäre Aufnahme in eine
        unserer Kliniken bitte die notwendigen Formulare, sewie den Einweisungsschein
        Ihres Arztes, die Krankenversicherungskarte und gültige Personalpapiere mit.'
        Leichte Sprache: 'Sie sind krank?
        Sie müssen ins Kranken-Haus zu einer Untersuchung? Oder zu einer Operation?
        Dann bringen Sie bitte diese Papiere mit ins Kranken-Haus:
        Den Zettel von Ihrem Haus-Arzt. Auf diesem Zettel steht, dass Sie ins Kranken-Haus müssen. Die Karte der Kranken-Versicherung. Ihren Ausweis.
        Bitte schauen Sie vorher in Ihrem Ausweis ein Datum nach.
        Unter dem Punkt „Ablaufdatum“ steht ein Datum mit einer Jahreszahl.
        Ist das dort angegebene Datum schon vorbei?
        Dann gilt dieser Ausweis nicht mehr.
        Sie müssen dann einen anderen Ausweis mitbringen,
        z. B. den Reisepass"""
        
        return self.prompt(system_prompt, input_text).text
    
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        system_prompt = """Rolle: Sprachassistent, zur Übersetzung normaler deutschen Text in Leichte Sprache.
        Input: 1 Satz in normaler deutschen Sprache. Output: 3 alternative übersetzte Sätze in Leichter Sprache. Output als Array von Strings. Striktes Output Format: ['suggestion1', 'suggestion2', 'suggestion3']
                Strikte Regeln:
                1. Verwende kurze Sätze. Jeder Satz enthält nur eine Aussage. 2. Schreibe
                nur Aktivsätze im Format Subjekt-Verb-Objekt. 3. Vermeide den Konjunktiv.
                Schreibe nur in der Wirklichkeitsform. 4. Ersetze den Genitiv durch präpositionale Fügungen mit „von“. 5. Nutze keine Synonyme oder Sonderzeichen.
                6. Formuliere Verneinungen positiv, wenn möglich. 7. Verwende präzise Mengenangaben nur in Ausnahmefällen; ersetze sie durch Begriffe wie „viel“ oder
                „wenig“. Ersetze Jahreszahlen durch Ausdrücke wie „vor langer Zeit“. 8. Verwende „Du“ und „Sie“ korrekt wie
                in der Standardsprache.
                Rechtschreibregeln: 1. Verwende Bindestriche, um Zusammensetzungen zu verdeutlichen (z. B. Welt-All). 2. Nutze keine
                durchgehenden Großbuchstaben oder Kursivschrift. 3. Schreibe jeden Satz in
                eine eigene Zeile.
                Regeln für den Inhalt: 1. Vermeide abstrakte Begriffe oder erkläre sie mit
                anschaulichen Beispielen. 2. Vermeide bildhafte Sprache. 3. Erkläre Fremd- und
                Fachwörter bei der ersten Erwähnung. 4. Schreibe Abkürzungen beim ersten
                Vorkommen aus.
                Beispiel: Normaler Text: 'Bringen Sie für Ihre stationäre Aufnahme in eine
                unserer Kliniken bitte die notwendigen Formulare, sewie den Einweisungsschein
                Ihres Arztes, die Krankenversicherungskarte und gültige Personalpapiere mit.'
                Leichte Sprache: 'Sie sind krank?
                Sie müssen ins Kranken-Haus zu einer Untersuchung? Oder zu einer Operation?
                Dann bringen Sie bitte diese Papiere mit ins Kranken-Haus:
                Den Zettel von Ihrem Haus-Arzt. Auf diesem Zettel steht, dass Sie ins Kranken-Haus müssen. Die Karte der Kranken-Versicherung. Ihren Ausweis.
                Bitte schauen Sie vorher in Ihrem Ausweis ein Datum nach.
                Unter dem Punkt „Ablaufdatum“ steht ein Datum mit einer Jahreszahl.
                Ist das dort angegebene Datum schon vorbei?
                Dann gilt dieser Ausweis nicht mehr.
                Sie müssen dann einen anderen Ausweis mitbringen,
                z. B. den Reisepass."""
        
        return self.prompt(system_prompt, input_text).text
    
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        system_prompt = """Rolle: Sprachassistent, zur Übersetzung normaler deutschen Text in Leichte Sprache.
        Input: 1 Satz in Leichter Sprache. Output: 3 alternative Sätze in Leichter Sprache. Output als Array von Strings. Striktes Output Format: ['suggestion1', 'suggestion2', 'suggestion3']
                Strikte Regeln:
                1. Verwende kurze Sätze. Jeder Satz enthält nur eine Aussage. 2. Schreibe
                nur Aktivsätze im Format Subjekt-Verb-Objekt. 3. Vermeide den Konjunktiv.
                Schreibe nur in der Wirklichkeitsform. 4. Ersetze den Genitiv durch präpositionale Fügungen mit „von“. 5. Nutze keine Synonyme oder Sonderzeichen.
                6. Formuliere Verneinungen positiv, wenn möglich. 7. Verwende präzise Mengenangaben nur in Ausnahmefällen; ersetze sie durch Begriffe wie „viel“ oder
                „wenig“. Ersetze Jahreszahlen durch Ausdrücke wie „vor langer Zeit“. 8. Verwende „Du“ und „Sie“ korrekt wie
                in der Standardsprache.
                Rechtschreibregeln: 1. Verwende Bindestriche, um Zusammensetzungen zu verdeutlichen (z. B. Welt-All). 2. Nutze keine
                durchgehenden Großbuchstaben oder Kursivschrift. 3. Schreibe jeden Satz in
                eine eigene Zeile.
                Regeln für den Inhalt: 1. Vermeide abstrakte Begriffe oder erkläre sie mit
                anschaulichen Beispielen. 2. Vermeide bildhafte Sprache. 3. Erkläre Fremd- und
                Fachwörter bei der ersten Erwähnung. 4. Schreibe Abkürzungen beim ersten
                Vorkommen aus.
                Beispiel: Normaler Text: 'Bringen Sie für Ihre stationäre Aufnahme in eine
                unserer Kliniken bitte die notwendigen Formulare, sewie den Einweisungsschein
                Ihres Arztes, die Krankenversicherungskarte und gültige Personalpapiere mit.'
                Leichte Sprache: 'Sie sind krank?
                Sie müssen ins Kranken-Haus zu einer Untersuchung? Oder zu einer Operation?
                Dann bringen Sie bitte diese Papiere mit ins Kranken-Haus:
                Den Zettel von Ihrem Haus-Arzt. Auf diesem Zettel steht, dass Sie ins Kranken-Haus müssen. Die Karte der Kranken-Versicherung. Ihren Ausweis.
                Bitte schauen Sie vorher in Ihrem Ausweis ein Datum nach.
                Unter dem Punkt „Ablaufdatum“ steht ein Datum mit einer Jahreszahl.
                Ist das dort angegebene Datum schon vorbei?
                Dann gilt dieser Ausweis nicht mehr.
                Sie müssen dann einen anderen Ausweis mitbringen,
                z. B. den Reisepass. """
        
        return self.prompt(system_prompt, input_text).text