from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer

from v2.dataset import create_dataloaders


def test_dataloader():
    sentences = [
        "זה הוא משפט בעברית המתאר חוויה אישית שקרתה לי אתמול בבוקר כאשר טיילתי בפארק",
        "אני אוהב ללמוד דברים חדשים, במיוחד כשזה קשור לטכנולוגיה ולתכנות מחשבים",
        "במהלך החופשה האחרונה שלנו נסענו לצפון הארץ ונהנינו מנופים יפים ומאוכל טעים",
        "היום בעבודה הייתי צריך להתמודד עם אתגר טכני מסובך אך הצלחתי לפתור אותו לבסוף",
        "בשבוע הבא מתוכנן אירוע משפחתי גדול ואני מאוד מצפה לפגוש את כל הקרובים שלי",
        "הלמידה מרחוק הפכה להיות חלק בלתי נפרד מהחיים שלנו בתקופה האחרונה",
        "אני מתכנן להשתתף במרתון הקרוב שיתקיים בעיר ולנסות לשפר את התוצאה האישית שלי",
        "בסוף השבוע אני מתכוון לנוח, לקרוא ספר טוב ולהתנתק מהשגרה היומיומית",
        "החברים שלי ואני מתכננים לצאת לקמפינג במדבר וליהנות מהכוכבים בלילה",
        "הפרויקט שאני עובד עליו בעבודה מתקדם יפה ואני מקווה לסיים אותו בזמן ולהציג אותו לצוות"
    ]
    he_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-big-he-en')
    tokenizer1 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-he-en')
    tokenizer2 = AutoTokenizer.from_pretrained('facebook/opt-350m')
    tokenizer3 = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-he')

    train_loader, eval_loader = create_dataloaders(sentences, he_en_model, tokenizer1, tokenizer2, tokenizer3, 8, 0.8)

    for batch in train_loader:
        for i in range(batch["input_ids_1"].size(0)):
            print("\nTokenizer 1 (Hebrew sentence minus last token):")
            input_ids_1 = batch["input_ids_1"][i]
            tokens_1 = tokenizer1.convert_ids_to_tokens(input_ids_1)
            for token, token_id in zip(tokens_1, input_ids_1):
                print(f"Token: {token}, Token ID: {token_id}")

            print("\nTokenizer 2 (English translation):")
            input_ids_2 = batch["input_ids_2"][i]
            tokens_2 = tokenizer2.convert_ids_to_tokens(input_ids_2)
            for token, token_id in zip(tokens_2, input_ids_2):
                print(f"Token: {token}, Token ID: {token_id}")

            print("\nTokenizer 3 (Full Hebrew sentence with start token):")
            input_ids_3 = batch["input_ids_3"][i]
            tokens_3 = tokenizer3.convert_ids_to_tokens(input_ids_3)
            for token, token_id in zip(tokens_3, input_ids_3):
                print(f"Token: {token}, Token ID: {token_id}")


# Example usage of the test function
test_dataloader()
