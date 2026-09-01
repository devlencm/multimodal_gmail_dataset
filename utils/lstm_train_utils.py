import numpy as np


def get_inter_session_split(user_ids, sessions, all_users, train_sessions):
    train_idx = []
    test_idx = []
    skipped_users = []

    for user in all_users:

        user_idx = np.where(user_ids == user)[0]
        user_sessions = np.sort(np.unique(sessions[user_idx]))

        # Select training sessions by ordinal position
        user_train_sessions = [
            user_sessions[sess - 1]
            for sess in train_sessions
            if sess - 1 < len(user_sessions)
        ]

        # Remaining sessions become the test set
        user_test_sessions = [
            s for s in user_sessions
            if s not in user_train_sessions
        ]

        # Skip users with no test sessions
        if len(user_test_sessions) == 0:
            skipped_users.append(
                {
                    "user": user,
                    "num_sessions": len(user_sessions),
                    "sessions": user_sessions
                }
            )
            continue

        # Construct train and test indices
        train_idx.extend(
            user_idx[np.isin(sessions[user_idx], user_train_sessions)]
        )

        test_idx.extend(
            user_idx[np.isin(sessions[user_idx], user_test_sessions)]
        )

    return train_idx, test_idx, skipped_users

def get_intra_session_split(user_ids, sessions, user, session, train_fraction=2 / 3):

    # Get the data for user/session combo
    user_session_idx = np.where(
        (user_ids == user) & (sessions == session)
    )[0]

    # Create split index
    n = len(user_session_idx)
    split = int(train_fraction * n)

    train_idx = user_session_idx[:split]
    test_idx = user_session_idx[split:]

    return train_idx, test_idx

def get_splits(
    user_ids,
    sessions,
    all_users,
    split_type,
    train_sessions,
    train_fraction=2 / 3
):
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    if split_type == "inter":

        # Build the per-user inter-session split first
        train_idx, test_idx, skipped_users = get_inter_session_split(
            user_ids,
            sessions,
            all_users,
            train_sessions
        )

        # Group the indices by user
        train_users = {
            user: np.asarray(train_idx)[user_ids[train_idx] == user]
            for user in all_users
        }

        test_users = {
            user: np.asarray(test_idx)[user_ids[test_idx] == user]
            for user in all_users
        }

        for user in all_users:

            # Target user's genuine training/testing data
            user_train_idx = train_users.get(user, [])
            user_test_idx = test_users.get(user, [])

            if not len(user_train_idx) or not len(user_test_idx):
                continue

            # Genuine target-user samples
            target_train_idx = list(user_train_idx)
            target_test_idx = list(user_test_idx)

            # Start training set with genuine samples
            train_idx_user = list(target_train_idx)

            # Add all other users as impostors
            for imp_user in all_users:

                if imp_user == user:
                    continue

                imp_train_idx = train_users.get(imp_user, [])

                if len(imp_train_idx):
                    train_idx_user.extend(list(imp_train_idx))

            # Add imposters
            test_idx_user = list(target_test_idx)

            for imp_user in all_users:

                if imp_user == user:
                    continue

                imp_test_idx = test_users.get(imp_user, [])

                if len(imp_test_idx):
                    test_idx_user.extend(list(imp_test_idx))

            yield (user, None, train_idx_user, test_idx_user)

    elif split_type == "intra":

        # Precompute chronological split for every user/session
        session_splits = {}

        for split_user in all_users:

            user_sessions = np.sort(
                np.unique(
                    sessions[user_ids == split_user]
                )
            )

            for split_session in user_sessions:

                train_idx, test_idx = get_intra_session_split(
                    user_ids,
                    sessions,
                    split_user,
                    split_session,
                    train_fraction
                )

                if len(train_idx) and len(test_idx):
                    session_splits[
                        (split_user, split_session)
                    ] = (
                        list(train_idx),
                        list(test_idx)
                    )

        # Select one fixed session per user to serve as that user's
        # impostor session for the entire experiment
        impostor_sessions = {}

        for imp_user in all_users:

            available_sessions = sorted(
                [
                    s
                    for s in np.unique(
                        sessions[user_ids == imp_user]
                    )
                    if (imp_user, s) in session_splits
                ]
            )

            if not available_sessions:
                continue

            impostor_sessions[imp_user] = rng.choice(
                available_sessions
            )

        for imp_user in sorted(impostor_sessions):
            print(
                f"User {imp_user}: "
                f"Session {impostor_sessions[imp_user]}"
            )

        # Create one-vs-rest split for each target user/session
        for user in all_users:

            user_sessions = np.sort(
                np.unique(
                    sessions[user_ids == user]
                )
            )

            for session in user_sessions:

                if (user, session) not in session_splits:
                    continue

                # Genuine target-user observations
                target_train_idx, target_test_idx = (
                    session_splits[(user, session)]
                )

                train_idx = list(target_train_idx)
                test_idx = list(target_test_idx)

                # Add the fixed impostor session from every other user
                for imp_user in all_users:

                    if imp_user == user:
                        continue

                    if imp_user not in impostor_sessions:
                        continue

                    imp_session = impostor_sessions[imp_user]

                    imp_train, imp_test = session_splits[
                        (imp_user, imp_session)
                    ]

                    train_idx.extend(imp_train)
                    test_idx.extend(imp_test)

                yield user, session, train_idx, test_idx