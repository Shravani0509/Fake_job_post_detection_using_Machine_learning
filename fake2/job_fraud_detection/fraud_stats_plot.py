import matplotlib.pyplot as plt

def plot_fraud_stats(total_posts, fraudulent_posts, non_fraudulent_posts):
    fraud_percentage = (fraudulent_posts / total_posts * 100) if total_posts > 0 else 0

    labels = ['Fraudulent Posts', 'Non-Fraudulent Posts']
    counts = [fraudulent_posts, non_fraudulent_posts]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts, color=['red', 'green'])

    ax.set_title('Job Fraud Detection Statistics')
    ax.set_ylabel('Number of Posts')
    ax.set_ylim(0, max(counts) + 10)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.figtext(0.9, 0.1, f'Total Posts Checked: {total_posts}\nFraud Percentage: {fraud_percentage:.2f}%', 
                horizontalalignment='right')

    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace these values with actual counts
    total_posts_checked = 100
    fraudulent_posts = 30
    non_fraudulent_posts = 70

    plot_fraud_stats(total_posts_checked, fraudulent_posts, non_fraudulent_posts)
