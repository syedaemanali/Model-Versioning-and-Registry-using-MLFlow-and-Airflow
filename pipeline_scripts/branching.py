def branching_logic(**context):
    ti = context['ti']
    accuracy = ti.xcom_pull(task_ids='evaluate_model', key='accuracy')

    print(f"Accuracy: {accuracy:.4f}")

    if accuracy >= 0.80:
        return 'register_model'
    else:
        return 'reject_model'