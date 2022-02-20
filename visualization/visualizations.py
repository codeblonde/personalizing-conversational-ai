"""Methods to visualize results and create knowledgeable graphs"""
import seaborn as sns
import matplotlib.pyplot as plt


# visualize traits in boxplot
def boxplot_trait_scores(df):
    """
    Plot boxplot diagramm of each trait in one comprehensive graph.
    """
    sns.set_theme(style='whitegrid')
    # palette = sns.color_palette('flare', as_cmap=True)  #rocket
    sns.set_palette('rocket_r')
    ax = sns.boxplot(data=df)
    # showmeans=True, meanprops={'markerfacecolor':'black', 'markeredgecolor':'black'})
    # Note: by default shows median not mean
    # To add mean set argument showmeans = True
    ax.set(xlabel='Personality Traits', ylabel='Expression Values')
    ax.set_title('Big Five Trait Expressions in Participant Sample')
    plt.show()


def histplot_messages(df):
    """
    Plot a histogramm of message/chat distribution per length and count
    """
    sns.set_theme(style='darkgrid')
    sns.set_palette('rocket_r')
    ax = sns.histplot(data=df, kde=True)
    plt.show()


def scatterplot_interaction(df):
    """
    Plot possible interaction between users' message lengths and extraversion scores
    """
    sns.set_theme(style='darkgrid')
    sns.set_palette('rocket')
    ax = sns.regplot(data=df, y='message', x='extraversion_scores')
    ax.set(xlabel='Scores for Trait Extraversion', ylabel='Length of Chat Messages')
    ax.set_title('Interaction between Extraversion Score and Message Length')
    plt.show()


def boxplot_lsm_scores(df):
    sns.set_theme(style='whitegrid')
    sns.set_palette('magma_r', n_colors=9)  # rocket
    ax = sns.boxplot(data=df.drop(columns=['chat_id', 'posemo', 'negemo'], axis=1))
    # showmeans=True, meanprops={'markerfacecolor':'black', 'markeredgecolor':'black'})
    ax.set(xlabel='Categories of Function Words', ylabel='LSM Score')
    ax.set_title('Degree of Linguistic Style Matching across Chats per Category')
    plt.show()
